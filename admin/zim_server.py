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
            namespace = getattr(article, 'namespace', '')

            if url:
                # Store multiple variations for flexible lookup
                url_index[url] = i

                # Store with explicit namespace prefix for assets
                if namespace and namespace not in ('A', ''):
                    prefixed_url = f"{namespace}/{url}"
                    if prefixed_url not in url_index:
                        url_index[prefixed_url] = i

                # Also store without namespace prefix (A/, I/, -, etc.)
                if '/' in url:
                    parts = url.split('/', 1)
                    # If first part looks like a namespace (single char or -)
                    if len(parts[0]) <= 1 or parts[0] == '-':
                        stripped = parts[1]
                        if stripped not in url_index:
                            url_index[stripped] = i
        except Exception:
            continue

    _url_index_cache[source_id] = url_index
    print(f"URL index built for {source_id}: {len(url_index)} entries")

    # Log some sample asset URLs for debugging
    asset_urls = [u for u in url_index.keys() if u.startswith('-/') or '.css' in u or '.js' in u]
    if asset_urls:
        print(f"[ZIM] Sample asset URLs in {source_id}: {asset_urls[:5]}")
    else:
        print(f"[ZIM] Warning: No asset URLs found in {source_id} - CSS/JS may not be available")


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
        f"-/{url_path}",  # Special namespace (assets/css/js)
        f"M/{url_path}",  # Metadata namespace
        url_path.replace('_', ' '),
        url_path.replace(' ', '_'),
    ]

    # If path starts with -/, it's already a namespace path - also try without the prefix
    # ZIM files store these as namespace "-" with url "mw/style.css"
    if url_path.startswith('-/'):
        inner_path = url_path[2:]  # Remove -/ prefix
        variations.extend([
            inner_path,
            f"-/{inner_path}",  # Explicitly with namespace
        ])

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
        # Debug: log failed lookups for assets (to help diagnose)
        if url_path.startswith('-/') or url_path.endswith('.css') or url_path.endswith('.js'):
            # Only log first few failures to avoid spam
            if not hasattr(_get_article_by_url, '_logged_failures'):
                _get_article_by_url._logged_failures = set()
            if url_path not in _get_article_by_url._logged_failures and len(_get_article_by_url._logged_failures) < 10:
                _get_article_by_url._logged_failures.add(url_path)
                print(f"[ZIM] Asset not found: {url_path} (tried {len(variations)} variations)")
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

    Also marks remaining external links with class="external-link" for styling.
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

    # Mark remaining external links (http:// or https://) with a class for styling
    # These are links that weren't matched by base_url rewrites
    # Add class="external-link" and data-external="true" for JS detection
    html_content = re.sub(
        r'<a\s+([^>]*?)href="(https?://[^"]+)"([^>]*)>',
        r'<a \1href="\2" class="external-link" data-external="true"\3>',
        html_content,
        flags=re.IGNORECASE
    )

    return html_content


def _inject_offline_banner(html_content: str, source_id: str, title: str = "", translation_lang: str = None) -> str:
    """
    Inject a minimal floating button for navigation back to search.
    Also injects CSS/JS for external link handling (strikethrough when offline).
    Designed to be unobtrusive - user won't notice unless they need it.

    Args:
        html_content: HTML to modify
        source_id: Source identifier
        title: Optional title (unused currently)
        translation_lang: Language code if content was translated, None otherwise
    """
    # CSS for external links - strikethrough by default (offline mode)
    # JS will remove strikethrough if online connectivity detected
    external_link_css = '''
<style id="clippy-external-link-styles">
/* External links: strikethrough when offline, normal when online */
a.external-link {
    text-decoration: line-through;
    color: #888;
    cursor: not-allowed;
    pointer-events: none;
}
a.external-link::after {
    content: " (offline)";
    font-size: 0.8em;
    color: #666;
}
/* When online, restore normal link behavior */
body.clippy-online a.external-link {
    text-decoration: underline;
    color: inherit;
    cursor: pointer;
    pointer-events: auto;
}
body.clippy-online a.external-link::after {
    content: "";
}
</style>
'''

    # JavaScript to check online status based on the app's connection mode
    # Not just internet connectivity - respects the app's offline_mode setting
    online_check_js = '''
<script id="clippy-online-check">
(function() {
    // Check the app's connection status (respects offline_mode setting)
    function checkOnlineStatus() {
        fetch('/api/v1/connection-status', {
            method: 'GET',
            cache: 'no-store'
        })
        .then(function(response) { return response.json(); })
        .then(function(data) {
            // Only enable external links if app is in online mode AND actually connected
            if (data.state === 'online' || data.state === 'degraded') {
                document.body.classList.add('clippy-online');
            } else {
                document.body.classList.remove('clippy-online');
            }
        })
        .catch(function() {
            // If we can't reach the local server, assume offline
            document.body.classList.remove('clippy-online');
        });
    }

    // Check on page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', checkOnlineStatus);
    } else {
        checkOnlineStatus();
    }

    // Re-check periodically (every 30 seconds)
    setInterval(checkOnlineStatus, 30000);
})();
</script>
'''

    # Translation indicator (shows when content was translated)
    translation_indicator = ""
    if translation_lang and translation_lang != "en":
        # Get language display name
        lang_names = {
            "es": "Spanish", "fr": "French", "ar": "Arabic", "zh": "Chinese",
            "pt": "Portuguese", "hi": "Hindi", "sw": "Swahili", "ht": "Haitian Creole"
        }
        lang_display = lang_names.get(translation_lang, translation_lang.upper())
        translation_indicator = f'''
<div id="clippy-translation-badge" style="
    position: fixed;
    bottom: 70px;
    right: 20px;
    background: rgba(78, 204, 163, 0.9);
    color: #1a1a2e;
    padding: 6px 12px;
    border-radius: 16px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 11px;
    z-index: 10000;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
">
    Translated to {lang_display}
</div>
'''

    # Minimal floating button in bottom-right corner
    banner_html = f'''
{external_link_css}
{online_check_js}
{translation_indicator}
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


def _translate_if_enabled(html_content: str, source_id: str, path: str) -> Tuple[str, Optional[str]]:
    """
    Translate HTML content if translation is enabled and a language pack is active.

    Phase 1: Translates article content from English to user's selected language.
    Uses TranslationService with caching for performance.

    Args:
        html_content: The HTML string to translate
        source_id: Source identifier (e.g., "appropedia")
        path: Article path within the source

    Returns:
        Tuple of (translated_html, language_code) where language_code is None if not translated
    """
    try:
        config = get_local_config()

        # Check if translation is enabled
        translation_enabled = config.is_translation_enabled()
        active_lang = config.get_translation_language()

        # Debug logging (only on first request or when settings change)
        if not hasattr(_translate_if_enabled, '_last_log') or \
           _translate_if_enabled._last_log != (translation_enabled, active_lang):
            _translate_if_enabled._last_log = (translation_enabled, active_lang)
            print(f"[Translation] Status: enabled={translation_enabled}, language={active_lang}")

        if not translation_enabled:
            return (html_content, None)

        if not active_lang or active_lang == "en":
            return (html_content, None)

        # Check if language pack is installed
        from offline_tools.language_registry import get_language_registry
        registry = get_language_registry()
        if not registry.is_pack_installed(active_lang):
            print(f"[Translation] Language pack not installed: {active_lang}")
            print(f"[Translation] Download the {active_lang} pack from Languages tab to enable auto-translation")
            return (html_content, None)

        # Create article ID for caching
        article_id = f"{source_id}/{path}"

        # Translate using singleton TranslationService (avoids reloading model)
        from offline_tools.translation import get_translation_service
        service = get_translation_service(active_lang)

        if not service.is_available():
            print(f"[Translation] Service not available for: {active_lang}")
            return (html_content, None)

        print(f"[Translation] Translating article to {active_lang}: {article_id}")
        translated = service.translate_html(html_content, article_id)
        return (translated, active_lang)

    except Exception as e:
        print(f"[Translation] Error translating content: {e}")
        return (html_content, None)


@router.get("/{source_id}/{path:path}")
async def serve_zim_content(source_id: str, path: str):
    """
    Serve content from a ZIM file.

    Example: /zim/bitcoin/wiki/Bitcoin -> serves the Bitcoin wiki page from bitcoin.zim
    """
    # Handle trailing slash: /zim/bitcoin/ becomes path="" - redirect to index
    if not path or path.strip() == "":
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/zim/{source_id}", status_code=307)

    zim = _get_zim_file(source_id)
    if zim is None:
        raise HTTPException(
            status_code=404,
            detail=f"ZIM source not found: {source_id}. Make sure the ZIM file exists in your backup folder."
        )

    # Use optimized lookup with URL index
    result = _get_article_by_url(source_id, zim, path)

    if result is None:
        # Return a friendly HTML error page instead of JSON error
        error_html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Article Not Found</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 600px;
            margin: 100px auto;
            padding: 20px;
            background: #1a1a2e;
            color: #e0e0e0;
            text-align: center;
        }}
        h1 {{ color: #f0ad4e; margin-bottom: 10px; }}
        .path {{ color: #4fc3f7; font-family: monospace; background: #16213e; padding: 8px 16px; border-radius: 4px; display: inline-block; margin: 10px 0; }}
        p {{ color: #aaa; line-height: 1.6; }}
        .back-btn {{
            display: inline-block;
            margin-top: 20px;
            padding: 12px 24px;
            background: #4fc3f7;
            color: #1a1a2e;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 500;
        }}
        .back-btn:hover {{ background: #81d4fa; }}
    </style>
</head>
<body>
    <h1>Article Not Found</h1>
    <div class="path">{path}</div>
    <p>This article isn't included in the <strong>{source_id}</strong> offline archive.</p>
    <p>ZIM archives only contain a subset of articles. Links to articles outside the archive won't work offline.</p>
    <a href="/" class="back-btn">Back to Search</a>
</body>
</html>'''
        return HTMLResponse(content=error_html, status_code=404)

    content, mimetype = result

    # For HTML content, rewrite links, translate if enabled, and add banner
    if 'text/html' in mimetype:
        try:
            html_str = content.decode('utf-8', errors='ignore')
            html_str = _rewrite_internal_links(html_str, source_id)

            # Phase 1: Translate article content if translation is enabled
            html_str, translation_lang = _translate_if_enabled(html_str, source_id, path)

            # Inject banner with translation indicator (if translated)
            html_str = _inject_offline_banner(html_str, source_id, translation_lang=translation_lang)
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

    # Try to find the main page - mainPage is an article INDEX, not a URL
    main_page_idx = zim.header_fields.get('mainPage', None)
    if main_page_idx is not None and main_page_idx != 4294967295:  # 4294967295 = no main page
        try:
            # Look up the article by index to get its URL
            article = zim.get_article_by_id(int(main_page_idx))
            if article:
                url = getattr(article, 'url', '')
                if url:
                    # Strip namespace prefix if present (A/, I/, etc.)
                    if '/' in url and len(url.split('/')[0]) <= 2:
                        url = url.split('/', 1)[1]
                    print(f"[ZIM] {source_id} mainPage index {main_page_idx} -> URL: {url}")
                    from fastapi.responses import RedirectResponse
                    return RedirectResponse(url=f"/zim/{source_id}/{url}")
        except Exception as e:
            print(f"[ZIM] Error looking up mainPage article: {e}")

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
    # Clear logged failures tracker
    if hasattr(_get_article_by_url, '_logged_failures'):
        _get_article_by_url._logged_failures = set()
    print("[ZIM] Cache cleared - indexes will be rebuilt on next access")
