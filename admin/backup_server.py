"""
HTML Backup Content Server
Serves content from HTML backup folders for offline browsing

Similar to zim_server.py but for HTML backups stored in:
    BACKUP_PATH/{source_id}/pages/*.html
    BACKUP_PATH/{source_id}/assets/*
"""

import re
import os
import json
import mimetypes
from pathlib import Path
from typing import Optional, Dict

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse

from admin.local_config import get_local_config
from offline_tools.schemas import get_manifest_file, get_backup_manifest_file, html_filename_to_title


router = APIRouter(prefix="/backup", tags=["backup"])


def _get_backup_folder() -> Optional[str]:
    """Get the backup folder path from config"""
    config = get_local_config()
    return config.get_backup_folder()


def _get_source_path(source_id: str) -> Optional[Path]:
    """
    Get the path to a source's backup folder.

    Returns path to: BACKUP_PATH/{source_id}/
    """
    backup_folder = _get_backup_folder()
    if not backup_folder:
        return None

    source_path = Path(backup_folder) / source_id
    if source_path.exists() and source_path.is_dir():
        return source_path

    return None


def _get_source_base_url(source_id: str) -> Optional[str]:
    """Get the base_url for a source from its manifest"""
    backup_folder = _get_backup_folder()
    if not backup_folder:
        return None

    manifest_path = Path(backup_folder) / source_id / get_manifest_file()
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            return manifest.get("base_url")
        except Exception:
            pass
    return None


def _find_file_in_source(source_path: Path, file_path: str) -> Optional[Path]:
    """
    Find a file within the source folder.

    Checks:
    1. pages/{file_path} - HTML content
    2. assets/{file_path} - CSS, JS, images
    3. {file_path} directly in source folder
    """
    # Normalize path separators
    file_path = file_path.replace('\\', '/')

    # Check pages folder first (for HTML files)
    pages_path = source_path / "pages" / file_path
    if pages_path.exists() and pages_path.is_file():
        return pages_path

    # Check assets folder (CSS, JS, images)
    assets_path = source_path / "assets" / file_path
    if assets_path.exists() and assets_path.is_file():
        return assets_path

    # Check directly in source folder
    direct_path = source_path / file_path
    if direct_path.exists() and direct_path.is_file():
        return direct_path

    # Try with .html extension if not found
    if not file_path.endswith('.html'):
        html_path = source_path / "pages" / f"{file_path}.html"
        if html_path.exists():
            return html_path

    return None


def _rewrite_internal_links(html_content: str, source_id: str, page_path: str = None) -> str:
    """
    Rewrite internal links in HTML content to point to our backup server.

    Converts:
    - href="/path" -> href="/backup/{source_id}/path"
    - src="/images/foo.png" -> src="/backup/{source_id}/images/foo.png"
    - href="https://originalsite.com/page" -> href="/backup/{source_id}/page" (if base_url matches)
    - src="image.jpg" -> src="/backup/{source_id}/virtual/path/image.jpg" (simple relative)

    Args:
        html_content: The HTML content to rewrite
        source_id: The backup source ID
        page_path: The flattened page filename (e.g., "Projects_PV_EnphasePV_Main.htm.html")
    """
    # Determine virtual directory from flattened page path
    # e.g., "Projects_PV_EnphasePV_Main.htm.html" -> "Projects/PV/EnphasePV"
    virtual_dir = ""
    if page_path:
        # Remove .html suffix (may have .htm.html)
        base_name = page_path
        if base_name.endswith('.html'):
            base_name = base_name[:-5]
        if base_name.endswith('.htm'):
            base_name = base_name[:-4]
        # Split by underscore and take all but last part (filename)
        parts = base_name.split('_')
        if len(parts) > 1:
            virtual_dir = '/'.join(parts[:-1])
    # Handle absolute URLs to the original site
    base_url = _get_source_base_url(source_id)
    if base_url:
        base_url = base_url.rstrip('/')

        # Handle common URL variations
        base_variations = [base_url]
        if base_url.startswith('https://'):
            base_variations.append(base_url.replace('https://', 'http://'))
        elif base_url.startswith('http://'):
            base_variations.append(base_url.replace('http://', 'https://'))

        # Add www variants
        for url in list(base_variations):
            if '://www.' in url:
                base_variations.append(url.replace('://www.', '://'))
            elif '://' in url and '://www.' not in url:
                base_variations.append(url.replace('://', '://www.'))

        # Rewrite absolute URLs matching the original site
        for base in base_variations:
            escaped_base = re.escape(base)
            # Match href="https://site.com/path" or src="https://site.com/path"
            html_content = re.sub(
                rf'(href|src)="{escaped_base}(/[^"]*)"',
                rf'\1="/backup/{source_id}\2"',
                html_content
            )
            # Match without path
            html_content = re.sub(
                rf'(href|src)="{escaped_base}"',
                rf'\1="/backup/{source_id}/"',
                html_content
            )

    # Rewrite absolute paths starting with /
    html_content = re.sub(
        r'(href|src)="(/[^"]*)"',
        rf'\1="/backup/{source_id}\2"',
        html_content
    )

    # Rewrite relative paths with ../ sequences: href="../something" or href="../../foo/bar"
    # Strip all leading ../ sequences and use just the remaining path
    def rewrite_relative_path(match):
        attr = match.group(1)  # href or src
        rel_path = match.group(2)  # The relative path like "../../../Styles/Site.css"
        # Strip all leading ../ or .. sequences
        clean_path = re.sub(r'^(\.\./)+', '', rel_path)
        # Also strip leading ./ if present
        clean_path = re.sub(r'^\./', '', clean_path)
        return f'{attr}="/backup/{source_id}/{clean_path}"'

    # Match href/src with relative paths starting with ../ or ./
    html_content = re.sub(
        r'(href|src)="((?:\.\./)+[^"]*)"',
        rewrite_relative_path,
        html_content
    )

    # Also handle ./ relative paths
    html_content = re.sub(
        r'(href|src)="(\./[^"]*)"',
        rewrite_relative_path,
        html_content
    )

    # Handle simple relative paths (no ./ or ../ prefix) like src="image.jpg"
    # These are relative to the original page's directory
    if virtual_dir:
        def rewrite_simple_relative(match):
            attr = match.group(1)  # href or src
            filename = match.group(2)  # Just the filename like "PanelL6.jpg"
            # Skip if it looks like an external URL or already absolute
            if filename.startswith(('http://', 'https://', '//', '/', '#', 'mailto:', 'tel:')):
                return match.group(0)
            # Skip data: URLs
            if filename.startswith('data:'):
                return match.group(0)
            return f'{attr}="/backup/{source_id}/{virtual_dir}/{filename}"'

        # Match simple relative paths - filename with extension, no path separators at start
        # Excludes paths starting with . / # or protocol
        html_content = re.sub(
            r'(href|src)="([^"./:#][^"]*\.[a-zA-Z0-9]+)"',
            rewrite_simple_relative,
            html_content
        )

    return html_content


def _inject_offline_banner(html_content: str, source_id: str) -> str:
    """
    Inject a minimal floating button for navigation back to search.
    """
    banner_html = f'''
<div id="clippy-nav-btn" style="
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgba(26, 26, 46, 0.9);
    border: 1px solid #4ecca3;
    border-radius: 8px;
    padding: 8px 16px;
    z-index: 99999;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 13px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
">
    <a href="/" style="color: #4ecca3; text-decoration: none;">
        << Back to Search
    </a>
</div>
'''
    # Inject before </body> tag
    body_end_match = re.search(r'</body>', html_content, re.IGNORECASE)
    if body_end_match:
        insert_pos = body_end_match.start()
        html_content = html_content[:insert_pos] + banner_html + html_content[insert_pos:]
    else:
        html_content = html_content + banner_html

    return html_content


def _get_mimetype(file_path: Path) -> str:
    """Get MIME type for a file"""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or 'application/octet-stream'


@router.get("/{source_id}/{path:path}")
async def serve_backup_content(source_id: str, path: str):
    """
    Serve content from an HTML backup folder.

    Example: /backup/builditsolar/Projects_Cooling_page.html
    """
    source_path = _get_source_path(source_id)
    if source_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Backup source not found: {source_id}. Make sure the folder exists in your backup path."
        )

    # Find the file
    file_path = _find_file_in_source(source_path, path)
    if file_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"File not found in backup: {path}"
        )

    # Read the file
    try:
        mimetype = _get_mimetype(file_path)

        # For HTML files, rewrite links and add navigation
        if 'text/html' in mimetype or file_path.suffix.lower() in ['.html', '.htm']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()

            html_content = _rewrite_internal_links(html_content, source_id, path)
            html_content = _inject_offline_banner(html_content, source_id)
            return HTMLResponse(content=html_content)

        # For other files (CSS, JS, images), serve directly
        with open(file_path, 'rb') as f:
            content = f.read()

        return Response(content=content, media_type=mimetype)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading file: {str(e)}"
        )


@router.get("/{source_id}")
async def serve_backup_index(source_id: str):
    """
    Serve the index page for an HTML backup.
    Tries to find index.html or shows a listing.
    """
    source_path = _get_source_path(source_id)
    if source_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Backup source not found: {source_id}"
        )

    # Try to find index.html
    index_file = _find_file_in_source(source_path, "index.html")
    if index_file:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/backup/{source_id}/index.html")

    # List available HTML files
    pages_dir = source_path / "pages"
    html_files = []
    if pages_dir.exists():
        html_files = sorted([f.name for f in pages_dir.glob("*.html")])[:50]  # Limit to 50

    # Get source name from manifest
    source_name = source_id
    manifest_path = source_path / get_manifest_file()
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            source_name = manifest.get("name", source_id)
        except Exception:
            pass

    # Generate index page
    files_html = ""
    for filename in html_files:
        display_name = html_filename_to_title(filename)
        files_html += f'<li><a href="/backup/{source_id}/{filename}">{display_name}</a></li>\n'

    if not files_html:
        files_html = "<li>No HTML files found</li>"

    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head>
    <title>{source_name} - Offline Backup</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #1a1a2e;
            color: #e0e0e0;
        }}
        h1 {{ color: #4ecca3; }}
        a {{ color: #4fc3f7; }}
        ul {{ line-height: 1.8; }}
        .back-link {{
            display: inline-block;
            margin-top: 20px;
            padding: 10px 20px;
            background: #16213e;
            border-radius: 4px;
            text-decoration: none;
        }}
    </style>
</head>
<body>
    <h1>{source_name}</h1>
    <p>Offline backup - {len(html_files)} pages available</p>
    <ul>
        {files_html}
    </ul>
    <a href="/" class="back-link"><< Back to Search</a>
</body>
</html>
""")


@router.get("/api/sources")
async def list_backup_sources():
    """List all available HTML backup sources"""
    backup_folder = _get_backup_folder()
    if not backup_folder:
        return {"sources": [], "error": "No backup folder configured"}

    backup_path = Path(backup_folder)
    sources = []

    for source_dir in backup_path.iterdir():
        if source_dir.is_dir():
            pages_dir = source_dir / "pages"
            if pages_dir.exists():
                # Count HTML files
                html_count = len(list(pages_dir.glob("*.html")))
                if html_count > 0:
                    sources.append({
                        "source_id": source_dir.name,
                        "page_count": html_count,
                        "type": "html"
                    })

    return {"sources": sources}
