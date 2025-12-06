"""
ZIM Content Server
Serves content from ZIM files for offline browsing
"""

import re
import os
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse

from admin.local_config import get_local_config
from offline_tools.schemas import get_manifest_file


router = APIRouter(prefix="/zim", tags=["zim"])

# Cache for ZIM file handles and URL indexes
_zim_cache: Dict[str, object] = {}
_url_index_cache: Dict[str, Dict[str, int]] = {}  # source_id -> {url: article_index}


def _get_backup_folder() -> Optional[str]:
    """Get the backup folder path from config"""
    config = get_local_config()
    return config.get_backup_folder()


def _find_zim_file(source_id: str) -> Optional[Path]:
    """
    Find the ZIM file for a given source ID.

    Looks in:
    1. BACKUP_PATH/{source_id}/*.zim (new structure)
    2. BACKUP_PATH/{source_id}.zim (legacy)
    """
    backup_folder = _get_backup_folder()
    if not backup_folder:
        return None

    backup_path = Path(backup_folder)

    # New structure: source folder with ZIM inside
    source_folder = backup_path / source_id
    if source_folder.exists() and source_folder.is_dir():
        zim_files = list(source_folder.glob("*.zim"))
        if zim_files:
            return zim_files[0]

    # Legacy: ZIM directly in backup folder
    legacy_zim = backup_path / f"{source_id}.zim"
    if legacy_zim.exists():
        return legacy_zim

    # Also check for partial match (source_id might be derived from filename)
    for zim_file in backup_path.glob("*.zim"):
        # Match if the filename starts with source_id
        if zim_file.stem.lower().startswith(source_id.lower()):
            return zim_file

    return None


def _get_zim_file(source_id: str):
    """
    Get or open a ZIM file for the given source.
    Uses caching to avoid reopening files.
    """
    global _zim_cache

    if source_id in _zim_cache:
        return _zim_cache[source_id]

    zim_path = _find_zim_file(source_id)
    if not zim_path:
        return None

    try:
        from zimply_core.zim_core import ZIMFile
        zim = ZIMFile(str(zim_path), 'utf-8')
        _zim_cache[source_id] = zim

        # Build URL index for fast lookups
        _build_url_index(source_id, zim)

        return zim
    except ImportError:
        print("zimply-core not installed. Run: pip install zimply-core")
        return None
    except Exception as e:
        print(f"Error opening ZIM file: {e}")
        return None


def _build_url_index(source_id: str, zim) -> None:
    """
    Build a URL to article index mapping for fast lookups.
    Called once when opening a ZIM file.
    """
    global _url_index_cache

    if source_id in _url_index_cache:
        return

    print(f"Building URL index for {source_id}...")
    url_index = {}
    article_count = zim.header_fields.get('articleCount', 0)

    for i in range(article_count):
        try:
            article = zim.get_article_by_id(i)
            if article is None:
                continue

            url = getattr(article, 'url', '')
            if url:
                # Store multiple variations for flexible lookup
                url_index[url] = i
                # Also store without namespace prefix (A/, I/, etc.)
                if '/' in url:
                    stripped = url.split('/', 1)[1]
                    if stripped not in url_index:
                        url_index[stripped] = i
        except Exception:
            continue

    _url_index_cache[source_id] = url_index
    print(f"URL index built for {source_id}: {len(url_index)} entries")


def _get_article_by_url(source_id: str, zim, url_path: str) -> Optional[Tuple[bytes, str]]:
    """
    Get article content from ZIM by URL path.
    Uses cached URL index for O(1) lookups.

    Returns:
        Tuple of (content_bytes, mimetype) or None if not found
    """
    global _url_index_cache

    # Normalize the URL path
    url_path = url_path.lstrip('/')

    # Get URL index for this source
    url_index = _url_index_cache.get(source_id, {})

    # Try various URL formats
    variations = [
        url_path,
        f"A/{url_path}",  # Article namespace
        f"I/{url_path}",  # Image namespace
        f"-/{url_path}",  # Special namespace
        url_path.replace('_', ' '),
        url_path.replace(' ', '_'),
    ]

    article_idx = None
    for var in variations:
        if var in url_index:
            article_idx = url_index[var]
            break

    if article_idx is None:
        # Fallback: try partial matching for paths with extensions
        base_path = url_path.rsplit('.', 1)[0] if '.' in url_path else url_path
        for stored_url, idx in url_index.items():
            if stored_url.endswith(url_path) or stored_url.endswith(base_path):
                article_idx = idx
                break

    if article_idx is None:
        return None

    try:
        article = zim.get_article_by_id(article_idx)
        if article is None:
            return None

        content = article.data
        mimetype = str(getattr(article, 'mimetype', 'text/html'))

        if isinstance(content, bytes):
            return (content, mimetype)
        elif isinstance(content, str):
            return (content.encode('utf-8'), mimetype)
    except Exception as e:
        print(f"Error reading article {article_idx}: {e}")
        return None

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


def _rewrite_internal_links(html_content: str, source_id: str) -> str:
    """
    Rewrite internal links in HTML content to point to our ZIM server.

    Converts:
    - href="/wiki/Page" -> href="/zim/{source_id}/wiki/Page"
    - href="../Page" -> href="/zim/{source_id}/Page"
    - src="/images/foo.png" -> src="/zim/{source_id}/images/foo.png"
    - href="https://deadsite.com/page" -> href="/zim/{source_id}/page" (if base_url matches)
    """
    # First, handle absolute URLs to the original site (for dead sites)
    # This allows seamless browsing even if the original site is gone
    base_url = _get_source_base_url(source_id)
    if base_url:
        # Normalize base_url (remove trailing slash)
        base_url = base_url.rstrip('/')

        # Also handle common variations (http vs https, www vs non-www)
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

        # Rewrite absolute URLs that match any base variation
        for base in base_variations:
            escaped_base = re.escape(base)
            # Match href="https://site.com/path" or src="https://site.com/path"
            html_content = re.sub(
                rf'(href|src)="{escaped_base}(/[^"]*)"',
                rf'\1="/zim/{source_id}\2"',
                html_content
            )
            # Also match without path (just the domain)
            html_content = re.sub(
                rf'(href|src)="{escaped_base}"',
                rf'\1="/zim/{source_id}/"',
                html_content
            )

    # Rewrite absolute paths starting with /
    # href="/something" -> href="/zim/source_id/something"
    html_content = re.sub(
        r'(href|src)="(/[^"]*)"',
        rf'\1="/zim/{source_id}\2"',
        html_content
    )

    # Rewrite relative paths that look like wiki links
    # href="PageName" or href="./PageName" (within same namespace)
    html_content = re.sub(
        r'href="(\./)?([A-Za-z0-9_:%-]+)"',
        rf'href="/zim/{source_id}/\2"',
        html_content
    )

    # Rewrite parent paths: href="../something"
    # This is tricky without knowing the current path, so we'll handle common cases
    html_content = re.sub(
        r'(href|src)="\.\.(/[^"]*)"',
        rf'\1="/zim/{source_id}\2"',
        html_content
    )

    return html_content


def _inject_offline_banner(html_content: str, source_id: str, title: str = "") -> str:
    """
    Inject a minimal floating button for navigation back to search.
    Designed to be unobtrusive - user won't notice unless they need it.
    """
    # Minimal floating button in bottom-right corner
    banner_html = f'''
<div id="clippy-nav-btn" style="
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgba(26, 26, 46, 0.9);
    color: #e0e0e0;
    padding: 10px 16px;
    border-radius: 24px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 13px;
    z-index: 10000;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 8px;
" onmouseover="this.style.background='rgba(79, 195, 247, 0.9)'"
   onmouseout="this.style.background='rgba(26, 26, 46, 0.9)'"
   onclick="window.location.href='/'">
    <span style="font-size: 16px;">&#8592;</span>
    <span>Back to Search</span>
</div>
'''

    # Try to inject before </body> tag for cleaner placement
    body_end_match = re.search(r'</body>', html_content, re.IGNORECASE)
    if body_end_match:
        insert_pos = body_end_match.start()
        html_content = html_content[:insert_pos] + banner_html + html_content[insert_pos:]
    else:
        # Append if no body tag found
        html_content = html_content + banner_html

    return html_content


@router.get("/{source_id}/{path:path}")
async def serve_zim_content(source_id: str, path: str):
    """
    Serve content from a ZIM file.

    Example: /zim/bitcoin/wiki/Bitcoin -> serves the Bitcoin wiki page from bitcoin.zim
    """
    zim = _get_zim_file(source_id)
    if zim is None:
        raise HTTPException(
            status_code=404,
            detail=f"ZIM source not found: {source_id}. Make sure the ZIM file exists in your backup folder."
        )

    # Use optimized lookup with URL index
    result = _get_article_by_url(source_id, zim, path)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Article not found in ZIM: {path}"
        )

    content, mimetype = result

    # For HTML content, rewrite links and add banner
    if 'text/html' in mimetype:
        try:
            html_str = content.decode('utf-8', errors='ignore')
            html_str = _rewrite_internal_links(html_str, source_id)
            html_str = _inject_offline_banner(html_str, source_id)
            return HTMLResponse(content=html_str)
        except Exception as e:
            print(f"Error processing HTML: {e}")
            return Response(content=content, media_type=mimetype)

    # For other content types (images, CSS, etc.), serve directly
    return Response(content=content, media_type=mimetype)


@router.get("/{source_id}")
async def serve_zim_index(source_id: str):
    """
    Serve the main page of a ZIM file.
    Redirects to the main page if available.
    """
    zim = _get_zim_file(source_id)
    if zim is None:
        raise HTTPException(
            status_code=404,
            detail=f"ZIM source not found: {source_id}"
        )

    # Try to find the main page
    main_page = zim.header_fields.get('mainPage', '')
    if main_page:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/zim/{source_id}/{main_page}")

    # Otherwise, return a simple index page
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head>
    <title>{source_id} - Offline Archive</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #1a1a2e;
            color: #e0e0e0;
        }}
        h1 {{ color: #4fc3f7; }}
        a {{ color: #4fc3f7; }}
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
    <h1>{source_id} - Offline Archive</h1>
    <p>This ZIM archive is available for offline browsing.</p>
    <p>Use the search on the main page to find specific articles.</p>
    <a href="/" class="back-link">Back to Search</a>
</body>
</html>
""")


@router.get("/api/sources")
async def list_zim_sources():
    """
    List all available ZIM sources.
    """
    backup_folder = _get_backup_folder()
    if not backup_folder:
        return {"sources": [], "error": "No backup folder configured"}

    backup_path = Path(backup_folder)
    sources = []

    # Find ZIM files in source folders
    for item in backup_path.iterdir():
        if item.is_dir():
            zim_files = list(item.glob("*.zim"))
            if zim_files:
                zim_file = zim_files[0]
                source_id = item.name

                # Try to get name from manifest
                manifest_path = item / get_manifest_file()
                name = source_id
                if manifest_path.exists():
                    try:
                        with open(manifest_path) as f:
                            manifest = json.load(f)
                            name = manifest.get("name", source_id)
                    except Exception:
                        pass

                sources.append({
                    "source_id": source_id,
                    "name": name,
                    "type": "zim",
                    "size_mb": round(zim_file.stat().st_size / (1024*1024), 2),
                    "browse_url": f"/zim/{source_id}"
                })

        # Legacy: ZIM files directly in backup folder
        elif item.suffix.lower() == '.zim':
            source_id = item.stem
            sources.append({
                "source_id": source_id,
                "name": source_id,
                "type": "zim",
                "size_mb": round(item.stat().st_size / (1024*1024), 2),
                "browse_url": f"/zim/{source_id}"
            })

    return {"sources": sources}


def cleanup_zim_cache():
    """Close all cached ZIM files and clear URL indexes"""
    global _zim_cache, _url_index_cache
    for source_id, zim in _zim_cache.items():
        try:
            zim.close()
        except Exception:
            pass
    _zim_cache = {}
    _url_index_cache = {}
