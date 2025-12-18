"""
PDF Content Server
Serves PDF files from source folders for browser viewing with page navigation.

Similar to backup_server.py but for PDF sources stored in:
    BACKUP_PATH/{source_id}/*.pdf
    BACKUP_PATH/{source_id}/pdfs/*.pdf

Browser handles #page=N navigation automatically when PDFs are served.
"""

import os
import json
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from admin.local_config import get_local_config
from offline_tools.schemas import get_manifest_file


router = APIRouter(prefix="/pdf", tags=["pdf"])


def _get_backup_folder() -> Optional[str]:
    """Get the backup folder path from config"""
    config = get_local_config()
    return config.get_backup_folder()


def _get_source_path(source_id: str) -> Optional[Path]:
    """
    Get the path to a source's folder.

    Returns path to: BACKUP_PATH/{source_id}/
    """
    backup_folder = _get_backup_folder()
    if not backup_folder:
        return None

    source_path = Path(backup_folder) / source_id
    if source_path.exists() and source_path.is_dir():
        return source_path

    return None


def _find_pdf_file(source_path: Path, filename: str) -> Optional[Path]:
    """
    Find a PDF file within the source folder.

    Checks:
    1. {source_path}/{filename} - directly in source folder
    2. {source_path}/pdfs/{filename} - in pdfs subfolder
    """
    # Normalize filename - strip any path traversal attempts
    filename = Path(filename).name

    # Check directly in source folder
    direct_path = source_path / filename
    if direct_path.exists() and direct_path.is_file() and direct_path.suffix.lower() == '.pdf':
        return direct_path

    # Check pdfs subfolder
    pdfs_path = source_path / "pdfs" / filename
    if pdfs_path.exists() and pdfs_path.is_file() and pdfs_path.suffix.lower() == '.pdf':
        return pdfs_path

    return None


def _get_pdf_files(source_path: Path) -> List[Path]:
    """Get all PDF files in a source folder"""
    pdfs = []

    # Check root folder
    pdfs.extend(source_path.glob("*.pdf"))

    # Check pdfs subfolder
    pdfs_folder = source_path / "pdfs"
    if pdfs_folder.exists():
        pdfs.extend(pdfs_folder.glob("*.pdf"))

    return sorted(pdfs, key=lambda p: p.name.lower())


@router.get("/{source_id}/{filename}")
async def serve_pdf(source_id: str, filename: str):
    """
    Serve a PDF file from a source folder.

    The browser handles #page=N navigation automatically.

    Example: /pdf/medical-guides/emergency_medicine.pdf#page=47
    """
    source_path = _get_source_path(source_id)
    if source_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"PDF source not found: {source_id}. Make sure the folder exists in your backup path."
        )

    # Find the PDF file
    pdf_path = _find_pdf_file(source_path, filename)
    if pdf_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"PDF not found: {filename}"
        )

    # Serve the PDF with correct MIME type
    # Browser will handle #page=N parameter for navigation
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name
    )


@router.get("/{source_id}")
async def serve_pdf_index(source_id: str):
    """
    Serve an index page listing available PDFs in a source.
    """
    source_path = _get_source_path(source_id)
    if source_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"PDF source not found: {source_id}"
        )

    # Get all PDFs
    pdf_files = _get_pdf_files(source_path)

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
    for pdf_path in pdf_files:
        display_name = pdf_path.stem.replace('_', ' ').replace('-', ' ').title()
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        files_html += f'<li><a href="/pdf/{source_id}/{pdf_path.name}">{display_name}</a> <span class="size">({size_mb:.1f} MB)</span></li>\n'

    if not files_html:
        files_html = "<li>No PDF files found</li>"

    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head>
    <title>{source_name} - PDF Collection</title>
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
        ul {{ line-height: 1.8; list-style: none; padding: 0; }}
        li {{ padding: 8px 0; border-bottom: 1px solid #2a2a4e; }}
        .size {{ color: #888; font-size: 0.9em; }}
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
    <p>PDF Collection - {len(pdf_files)} documents available</p>
    <ul>
        {files_html}
    </ul>
    <a href="/" class="back-link">&lt;&lt; Back to Search</a>
</body>
</html>
""")


@router.get("/api/sources")
async def list_pdf_sources():
    """List all available PDF sources"""
    backup_folder = _get_backup_folder()
    if not backup_folder:
        return {"sources": [], "error": "No backup folder configured"}

    backup_path = Path(backup_folder)
    sources = []

    for source_dir in backup_path.iterdir():
        if source_dir.is_dir():
            pdf_files = _get_pdf_files(source_dir)
            if pdf_files:
                sources.append({
                    "source_id": source_dir.name,
                    "pdf_count": len(pdf_files),
                    "type": "pdf"
                })

    return {"sources": sources}
