"""
Offline Content Server Routes

Serves offline content for preview in Source Tools wizard.
Handles both HTML scrapes (files in pages/ folder) and ZIM imports (content from ZIM file).

Source types handled:
- ZIM imports: HTML in pages/, assets served from ZIM file on-demand
- HTML scrapes: HTML in pages/, assets in extracted folders
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, Response
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json
import mimetypes
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search-test", tags=["offline-content"])

# Simple LRU-style cache for ZIM file handles to avoid reopening for every asset
_zim_cache: Dict[str, Any] = {}
_zim_cache_max = 3  # Keep max 3 ZIM files open


def get_backup_folder() -> Optional[Path]:
    """Get the backup folder path"""
    try:
        from admin.local_config import get_local_config
        config = get_local_config()
        folder = config.get_backup_folder()
        return Path(folder) if folder else None
    except Exception:
        return None


def get_source_manifest(source_id: str) -> Optional[Dict[str, Any]]:
    """Load manifest for a source - checks both _manifest.json and manifest.json"""
    backup = get_backup_folder()
    if not backup:
        return None

    # Try _manifest.json first (standard), then manifest.json (legacy/ZIM import)
    manifest_path = backup / source_id / "_manifest.json"
    if not manifest_path.exists():
        manifest_path = backup / source_id / "manifest.json"
        if not manifest_path.exists():
            return None

    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def get_cached_zim(zim_path: str):
    """Get a cached ZIM file handle, opening if needed."""
    global _zim_cache

    if zim_path in _zim_cache:
        return _zim_cache[zim_path]

    try:
        from zimply_core.zim_core import ZIMFile
        zim = ZIMFile(zim_path, 'utf-8')

        # Evict oldest if cache full
        if len(_zim_cache) >= _zim_cache_max:
            oldest_key = next(iter(_zim_cache))
            try:
                _zim_cache[oldest_key].close()
            except Exception:
                pass
            del _zim_cache[oldest_key]

        _zim_cache[zim_path] = zim
        return zim
    except ImportError:
        logger.error("zimply-core not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to open ZIM file {zim_path}: {e}")
        return None


# Cache for URL indexes (zim_path -> {url: article_index})
_url_index_cache: Dict[str, Dict[str, int]] = {}

# Cache for WARC ZIM domains (zim_path -> [domain1, domain2, ...])
_warc_domains_cache: Dict[str, List[str]] = {}

# Cache for ZIM type detection (zim_path -> is_warc)
_zim_type_cache: Dict[str, bool] = {}


def _get_index_cache_path(zim_path: str) -> Path:
    """Get path for persisted URL index cache file."""
    return Path(zim_path).parent / "_zim_url_index.json"


def build_url_index(zim_path: str, zim) -> Dict[str, int]:
    """
    Build a URL to article index mapping for fast lookups.
    Index is persisted to disk so it only needs to be built once per ZIM.
    """
    global _url_index_cache

    # Check in-memory cache first
    if zim_path in _url_index_cache:
        return _url_index_cache[zim_path]

    # Check for persisted index on disk
    cache_path = _get_index_cache_path(zim_path)
    if cache_path.exists():
        try:
            import json
            with open(cache_path, 'r') as f:
                url_index = json.load(f)
            _url_index_cache[zim_path] = url_index
            logger.info(f"Loaded URL index from disk: {len(url_index)} entries")
            return url_index
        except Exception as e:
            logger.warning(f"Failed to load cached index, rebuilding: {e}")

    # Build index from ZIM (slow, only done once)
    logger.info(f"Building URL index for {zim_path} (this may take several minutes for large ZIMs)...")
    url_index = {}

    try:
        header = getattr(zim, 'header_fields', {})
        article_count = header.get('articleCount', 0)

        for i in range(article_count):
            try:
                article = zim.get_article_by_id(i)
                if article is None:
                    continue

                url = getattr(article, 'url', '') or ''
                namespace = getattr(article, 'namespace', '') or ''

                if url:
                    # Store the URL as-is (handles "C" namespace format)
                    url_index[url] = i

                    # Also store with namespace prefix for compatibility
                    if namespace and namespace not in ('', 'C'):
                        prefixed_url = f"{namespace}/{url}"
                        if prefixed_url not in url_index:
                            url_index[prefixed_url] = i

                    # Store URL without any leading namespace-like prefix
                    if '/' in url:
                        parts = url.split('/', 1)
                        if len(parts[0]) <= 2:  # Single char namespace like A, I, -, or C
                            stripped = parts[1]
                            if stripped not in url_index:
                                url_index[stripped] = i
            except Exception:
                continue

        # Save to disk for future use
        try:
            import json
            with open(cache_path, 'w') as f:
                json.dump(url_index, f)
            logger.info(f"URL index saved to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save URL index to disk: {e}")

        _url_index_cache[zim_path] = url_index
        logger.info(f"URL index built: {len(url_index)} entries")
    except Exception as e:
        logger.error(f"Failed to build URL index: {e}")
        _url_index_cache[zim_path] = {}

    return _url_index_cache.get(zim_path, {})


def is_warc_style_zim(zim_path: str, url_index: Dict[str, int]) -> bool:
    """
    Detect if a ZIM is a WARC-style web archive.

    WARC ZIMs (created by tools like Browsertrix) have:
    - _zim_static/wombat.js (web archive replay library)
    - Assets stored with original domain paths (www.fema.gov/..., etc.)

    OpenZIM/Wikipedia ZIMs have:
    - _mw_/, _assets_/, I/, -/ namespace prefixes
    """
    global _zim_type_cache

    if zim_path in _zim_type_cache:
        return _zim_type_cache[zim_path]

    # Check for WARC signature: _zim_static/wombat.js
    is_warc = "_zim_static/wombat.js" in url_index or "_zim_static/wombatSetup.js" in url_index

    _zim_type_cache[zim_path] = is_warc
    if is_warc:
        logger.info(f"Detected WARC-style ZIM: {zim_path}")

    return is_warc


def get_warc_domains(zim_path: str, url_index: Dict[str, int]) -> List[str]:
    """
    Extract list of domains from a WARC ZIM URL index.

    Returns domains sorted by frequency (most common first) for faster matching.
    """
    global _warc_domains_cache
    import re

    if zim_path in _warc_domains_cache:
        return _warc_domains_cache[zim_path]

    domain_counts: Dict[str, int] = {}
    domain_pattern = re.compile(r'^([a-zA-Z0-9.-]+\.[a-z]{2,})/')

    for url in url_index.keys():
        match = domain_pattern.match(url)
        if match:
            domain = match.group(1)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

    # Sort by frequency (most common domains first for faster matching)
    sorted_domains = sorted(domain_counts.keys(), key=lambda d: domain_counts[d], reverse=True)

    _warc_domains_cache[zim_path] = sorted_domains
    logger.info(f"Found {len(sorted_domains)} domains in WARC ZIM: {sorted_domains[:5]}...")

    return sorted_domains


def generate_warc_url_variations(asset_url: str, domains: List[str]) -> List[str]:
    """
    Generate URL variations for WARC-style ZIMs.

    WARC ZIMs store assets with their original domain paths:
    - www.fema.gov/profiles/femad8_gov/themes/...
    - www.dhs.gov/profiles/dhsd8_gov/modules/...

    When HTML requests /profiles/xxx/..., we need to try prepending each domain.
    """
    urls = [asset_url]  # Always try as-is first

    # Handle _zim_static paths (web archive replay library)
    if "wombat" in asset_url.lower() or asset_url.startswith("_zim_static/"):
        urls.append(f"_zim_static/{asset_url}")
        urls.append(f"_zim_static/wombat.js")
        urls.append(f"_zim_static/wombatSetup.js")
        urls.append(f"_zim_static/__wb_module_decl.js")

    # Strip leading slash for consistent handling
    clean_url = asset_url.lstrip('/')
    urls.append(clean_url)

    # Try prepending each domain to the path
    for domain in domains:
        urls.append(f"{domain}/{clean_url}")

        # Also try without www. prefix if present, or with it if not
        if domain.startswith("www."):
            alt_domain = domain[4:]
            urls.append(f"{alt_domain}/{clean_url}")
        else:
            alt_domain = f"www.{domain}"
            urls.append(f"{alt_domain}/{clean_url}")

    # For CDN resources, try common CDN paths
    if asset_url.endswith(('.js', '.css', '.woff', '.woff2', '.ttf')):
        for domain in ['cdn.jsdelivr.net', 'fonts.googleapis.com', 'fonts.gstatic.com']:
            if domain in domains:
                urls.append(f"{domain}/{clean_url}")

    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls


def generate_zim_url_variations(asset_url: str) -> List[str]:
    """
    Generate possible ZIM URL variations for an asset path.

    ZIM files from different sources use different conventions:
    - Wikipedia/Kiwix (OpenZIM format):
      - Articles: A/Article_Name
      - Resources: -/mw/foo.css, -/res/foo.css
      - Images: I/hash/image.png or I/m/hash/image.png

    The HTML extracted from ZIM uses paths like:
      - /_mw_/foo.css (becomes _mw_/foo.css in our requests)
      - /_res_/foo.css
      - /_assets_/hash/image.png
    """
    urls = [asset_url]  # Always try as-is first

    # Handle underscore-prefixed paths (Kiwix Wikipedia ZIMs)
    if asset_url.startswith('_'):
        parts = asset_url.split('/', 1)
        if len(parts) == 2:
            prefix = parts[0]  # _mw_, _res_, etc.
            namespace = prefix.strip('_')  # mw, res, etc.
            rest = parts[1]

            # Standard OpenZIM mapping: _mw_/foo -> -/mw/foo
            urls.append(f"-/{namespace}/{rest}")
            urls.append(f"-/{rest}")
            urls.append(rest)
            urls.append(f"A/{namespace}/{rest}")
            urls.append(f"M/{rest}")

    # For _assets_ paths (images in Wikipedia ZIMs)
    # Format: _assets_/hash/filename.ext
    if asset_url.startswith('_assets_/'):
        img_path = asset_url.replace('_assets_/', '', 1)
        urls.append(f"I/{img_path}")
        urls.append(f"I/m/{img_path}")

        if '/' in img_path:
            parts = img_path.split('/')
            hash_folder = parts[0]  # The hash like 0c70a452f799bfe840676ee341124611
            filename_only = parts[-1]

            # Try various combinations
            urls.append(f"I/{filename_only}")
            urls.append(f"I/m/{filename_only}")
            urls.append(f"I/{hash_folder}/{filename_only}")
            urls.append(f"I/m/{hash_folder}/{filename_only}")

            # Some ZIMs use A/ namespace for images
            urls.append(f"A/{img_path}")
            urls.append(f"A/{filename_only}")

            # Try without namespace prefix
            urls.append(f"-/{img_path}")
            urls.append(img_path)
            urls.append(filename_only)

            # Try with just the hash as folder
            urls.append(f"{hash_folder}/{filename_only}")

    # For _res_ paths (resources - CSS, etc.)
    if asset_url.startswith('_res_/'):
        res_path = asset_url.replace('_res_/', '', 1)
        urls.append(f"-/{res_path}")
        urls.append(f"-/res/{res_path}")
        urls.append(f"-/style/{res_path}")
        urls.append(res_path)

    # For _webp_ paths (WebP handler JS)
    if asset_url.startswith('_webp_/'):
        webp_path = asset_url.replace('_webp_/', '', 1)
        urls.append(f"-/{webp_path}")
        urls.append(f"-/webp/{webp_path}")
        urls.append(f"-/js/{webp_path}")
        urls.append(webp_path)

    # For _mw_ paths specifically (MediaWiki assets)
    if asset_url.startswith('_mw_/'):
        mw_path = asset_url.replace('_mw_/', '', 1)
        urls.append(f"-/{mw_path}")
        urls.append(f"-/mw/{mw_path}")
        urls.append(f"-/s/{mw_path}")
        urls.append(f"-/skin/{mw_path}")
        urls.append(f"-/skins/{mw_path}")
        if mw_path.endswith('.svg'):
            urls.append(f"I/{mw_path}")
            urls.append(f"I/m/{mw_path}")

    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls


def serve_zim_asset(zim_path: str, asset_url: str) -> Optional[Tuple[bytes, Optional[str]]]:
    """
    Serve an asset directly from a ZIM file.
    Returns (content_bytes, mimetype) or None if not found.

    Automatically detects ZIM type:
    - WARC-style: Web archives with original domain paths
    - OpenZIM-style: Wikipedia/Kiwix with namespace prefixes

    Uses indexed lookup for reliability with all ZIM formats including
    newer "C" namespace format from Kiwix.
    """
    zim = get_cached_zim(zim_path)
    if not zim:
        return None

    # Build URL index (cached after first call)
    url_index = build_url_index(zim_path, zim)

    # Detect ZIM type and generate appropriate URL variations
    if is_warc_style_zim(zim_path, url_index):
        # WARC-style: try domain-prefixed paths
        domains = get_warc_domains(zim_path, url_index)
        urls_to_try = generate_warc_url_variations(asset_url, domains)
    else:
        # OpenZIM/Wikipedia: use namespace-based variations
        urls_to_try = generate_zim_url_variations(asset_url)

    logger.debug(f"Looking for ZIM asset: {asset_url} (trying {len(urls_to_try)} variations)")

    # First try indexed lookup (fast, O(1))
    article_idx = None
    matched_url = None
    for url in urls_to_try:
        if url in url_index:
            article_idx = url_index[url]
            matched_url = url
            break

    # If not found in index, try direct get_article_by_url as fallback
    if article_idx is None:
        for url in urls_to_try:
            try:
                article = zim.get_article_by_url(url)
                if article is not None:
                    content = article.data
                    mimetype = str(getattr(article, 'mimetype', '')) or None

                    if isinstance(content, str):
                        content = content.encode('utf-8')

                    logger.debug(f"Found ZIM asset via direct lookup: {url}")
                    return (content, mimetype)
            except Exception:
                continue
    else:
        # Retrieve by index
        try:
            article = zim.get_article_by_id(article_idx)
            if article is not None:
                content = article.data
                mimetype = str(getattr(article, 'mimetype', '')) or None

                if isinstance(content, str):
                    content = content.encode('utf-8')

                logger.debug(f"Found ZIM asset via index: {matched_url} -> idx {article_idx}")
                return (content, mimetype)
        except Exception as e:
            logger.debug(f"Failed to retrieve indexed article {article_idx}: {e}")

    # DEBUG level - missing assets are expected for mini ZIMs with inlined CSS/JS
    # Empty placeholders are returned to prevent console errors
    logger.debug(f"ZIM asset not found: {asset_url} (tried {len(urls_to_try)} paths)")
    return None


# ZIM asset path prefixes that indicate content should come from ZIM file
# OpenZIM (Wikipedia/Kiwix): _mw_/, _res_/, _assets_/, _webp_/, -/, I/, M/
# WARC (web archives): _zim_static/
ZIM_ASSET_PREFIXES = ('_mw_/', '_res_/', '_assets_/', '_webp_/', '-/', 'I/', 'M/', '_zim_static/')


@router.get("/content/{source_id}/{filename:path}")
async def serve_content(source_id: str, filename: str):
    """
    Serve offline content from a source.

    For HTML pages: serves from pages/ folder with base tag injection.
    For ZIM assets (CSS, JS, images): serves directly from ZIM file.
    For HTML scrape assets: serves from extracted folders.
    """
    import urllib.parse

    filename = urllib.parse.unquote(filename)

    backup = get_backup_folder()
    if not backup:
        raise HTTPException(500, "Backup folder not configured")

    source_path = backup / source_id
    pages_path = source_path / "pages"

    manifest = get_source_manifest(source_id)
    is_zim_source = manifest and manifest.get("created_from") == "zim_import"

    # For ZIM sources with asset paths, serve directly from ZIM file
    if is_zim_source and any(filename.startswith(p) for p in ZIM_ASSET_PREFIXES):
        zim_path = manifest.get("zim_path")
        if not zim_path:
            raise HTTPException(404, "ZIM file path not in manifest")

        if not Path(zim_path).exists():
            raise HTTPException(404, f"ZIM file not found: {zim_path}")

        result = serve_zim_asset(zim_path, filename)
        if result:
            content, mimetype = result
            if not mimetype:
                mimetype, _ = mimetypes.guess_type(filename)
                if not mimetype:
                    mimetype = "application/octet-stream"
            return Response(content=content, media_type=mimetype)
        else:
            # Return empty placeholders for missing CSS/JS to prevent console errors
            if filename.endswith('.css'):
                return Response(content=b"/* Asset not in ZIM */", media_type="text/css")
            elif filename.endswith('.js'):
                return Response(content=b"/* Asset not in ZIM */", media_type="application/javascript")
            else:
                raise HTTPException(404, f"Asset not found in ZIM: {filename}")

    # Security: ensure path doesn't escape source folder
    try:
        file_path = (pages_path / filename).resolve()
        if not str(file_path).startswith(str(source_path.resolve())):
            raise HTTPException(403, "Access denied")
    except Exception:
        raise HTTPException(400, "Invalid path")

    # For HTML scrapes, assets might be in source root (not in pages/)
    # e.g., builditsolar2/assets/ instead of builditsolar2/pages/assets/
    if not file_path.exists():
        alt_path = (source_path / filename).resolve()
        if str(alt_path).startswith(str(source_path.resolve())) and alt_path.exists():
            file_path = alt_path

    # For HTML scrapes, internal links use original paths (e.g., ../index.htm)
    # but scraped files are in pages/ with different naming conventions:
    # 1. Simple: index.htm -> pages/index.htm.html
    # 2. Path-encoded: GettingStarted/GettingStarted.htm -> pages/GettingStarted_GettingStarted.htm.html
    if not file_path.exists() and not is_zim_source:
        # Try 1: Direct path with .html extension added
        pages_filename = f"{filename}.html" if not filename.endswith('.html') else filename
        alt_path = (pages_path / pages_filename).resolve()
        if str(alt_path).startswith(str(source_path.resolve())) and alt_path.exists():
            file_path = alt_path
        else:
            # Try 2: Replace path separators with underscores (common scraper pattern)
            # e.g., GettingStarted/GettingStarted.htm -> GettingStarted_GettingStarted.htm.html
            flat_filename = filename.replace('/', '_').replace('\\', '_')
            if not flat_filename.endswith('.html'):
                flat_filename = f"{flat_filename}.html"
            alt_path = (pages_path / flat_filename).resolve()
            if str(alt_path).startswith(str(source_path.resolve())) and alt_path.exists():
                file_path = alt_path
            else:
                # Try 3: Just the base filename with .html in pages/
                base_name = Path(filename).name
                alt_path = (pages_path / f"{base_name}.html").resolve()
                if str(alt_path).startswith(str(source_path.resolve())) and alt_path.exists():
                    file_path = alt_path

    if not file_path.exists():
        # For ZIM sources, try serving from ZIM as a fallback
        if is_zim_source:
            zim_path = manifest.get("zim_path")
            if zim_path and Path(zim_path).exists():
                result = serve_zim_asset(zim_path, filename)
                if result:
                    content, mimetype = result
                    if not mimetype:
                        mimetype, _ = mimetypes.guess_type(filename)
                        if not mimetype:
                            mimetype = "application/octet-stream"
                    return Response(content=content, media_type=mimetype)
                else:
                    # Return empty placeholders for missing CSS/JS (common in web archives)
                    if filename.endswith('.css'):
                        return Response(content=b"/* Asset not in archive */", media_type="text/css")
                    elif filename.endswith('.js'):
                        return Response(content=b"/* Asset not in archive */", media_type="application/javascript")
        raise HTTPException(404, f"File not found: {filename}")

    content_type, _ = mimetypes.guess_type(str(file_path))
    if not content_type:
        content_type = "text/html" if file_path.suffix == ".html" else "application/octet-stream"

    # For HTML files, inject base tag and offline indicator
    if content_type == "text/html":
        try:
            import re
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            base_path = f"/useradmin/search-test/content/{source_id}/"

            # Rewrite absolute asset paths to relative paths so base tag works
            # ZIM HTML uses absolute paths like /_mw_/foo.css, /_assets_/img.png
            # Convert to relative: _mw_/foo.css so base tag redirects them correctly
            asset_prefixes = ['_mw_', '_res_', '_assets_', '_webp_']
            for prefix in asset_prefixes:
                # Match href="/_prefix_/..." or src="/_prefix_/..."
                content = re.sub(
                    rf'(href|src)=(["\'])/({prefix}/)',
                    rf'\1=\2\3',
                    content,
                    flags=re.IGNORECASE
                )

            # Rewrite relative paths to absolute paths with full route prefix
            # This is more reliable than using base tag which can be overridden
            # Convert: ../I/foo.jpg -> /useradmin/search-test/content/source_id/I/foo.jpg
            # Handles any number of ../ prefixes

            # Pattern for OpenZIM paths (../I/, ../-/, etc.)
            content = re.sub(
                r'(href|src)=(["\'])(?:\.\./)+([-IM]/)',
                rf'\1=\2{base_path}\3',
                content,
                flags=re.IGNORECASE
            )

            # Pattern for WARC files with extension (../../../../_zim_static/wombat.js, etc.)
            content = re.sub(
                r'(href|src)=(["\'])(?:\.\./)+([-_a-zA-Z0-9]+\.[a-zA-Z0-9]+)',
                rf'\1=\2{base_path}\3',
                content,
                flags=re.IGNORECASE
            )

            # Pattern for WARC paths with folders (../../../sites/default/..., ../../../profiles/..., etc.)
            content = re.sub(
                r'(href|src)=(["\'])(?:\.\./)+([a-zA-Z0-9._-]+/)',
                rf'\1=\2{base_path}\3',
                content,
                flags=re.IGNORECASE
            )

            # Pattern for HTML scrape internal links (../index.htm, ../GettingStarted/GettingStarted.htm)
            # Matches: ../ prefixes followed by optional subdirs and a filename with extension
            content = re.sub(
                r'(href|src)=(["\'])(?:\.\./)+([a-zA-Z0-9._-]+(?:/[a-zA-Z0-9._-]+)*\.[a-zA-Z0-9]+)',
                rf'\1=\2{base_path}\3',
                content,
                flags=re.IGNORECASE
            )

            # Also inject base tag as fallback for any paths we missed
            if '<base' not in content.lower():
                head_insert = f'<base href="{base_path}">'
                if '<head>' in content.lower():
                    content = content.replace('<head>', f'<head>\n{head_insert}', 1)
                    content = content.replace('<HEAD>', f'<HEAD>\n{head_insert}', 1)
                else:
                    content = f'{head_insert}\n{content}'

            # Add offline indicator and external link handling
            indicator = '''
<style>
.offline-indicator {
    position: fixed;
    top: 10px;
    right: 10px;
    background: #1a1a2e;
    color: #4ecca3;
    padding: 5px 10px;
    border-radius: 4px;
    font-family: sans-serif;
    font-size: 12px;
    z-index: 99999;
    border: 1px solid #4ecca3;
}
/* External links: strikethrough in offline preview */
a.clippy-external {
    text-decoration: line-through;
    color: #888 !important;
    cursor: not-allowed;
    pointer-events: none;
}
a.clippy-external::after {
    content: " [offline]";
    font-size: 0.75em;
    color: #666;
}
/* Internal links that are missing from backup */
a.clippy-missing {
    text-decoration: line-through;
    color: #c44 !important;
    cursor: not-allowed;
    pointer-events: none;
}
a.clippy-missing::after {
    content: " [not in backup]";
    font-size: 0.75em;
    color: #a33;
}
</style>
<script>
(function() {
    function markLinks() {
        var links = document.querySelectorAll('a[href]');
        var internalLinks = [];
        for (var i = 0; i < links.length; i++) {
            var href = links[i].getAttribute('href');
            if (!href || href.startsWith('#') || href.startsWith('javascript:')) continue;
            // External = starts with http:// or https:// and not our local server
            if (href.match(/^https?:\/\//i) && href.indexOf(window.location.host) === -1) {
                links[i].classList.add('clippy-external');
                links[i].title = 'External link - not available offline';
            } else if (!href.match(/^https?:\/\//i) && !href.startsWith('mailto:')) {
                // Internal link - queue for availability check
                internalLinks.push(links[i]);
            }
        }
        // Check internal links (batch with delay to not spam server)
        var checkIdx = 0;
        function checkNextBatch() {
            var batch = internalLinks.slice(checkIdx, checkIdx + 5);
            if (batch.length === 0) return;
            batch.forEach(function(link) {
                var href = link.getAttribute('href');
                // Resolve relative URL
                var url = new URL(href, window.location.href).href;
                fetch(url, { method: 'HEAD' })
                    .then(function(resp) {
                        if (!resp.ok) {
                            link.classList.add('clippy-missing');
                            link.title = 'Page not in offline backup';
                        }
                    })
                    .catch(function() {
                        link.classList.add('clippy-missing');
                        link.title = 'Page not in offline backup';
                    });
            });
            checkIdx += 5;
            if (checkIdx < internalLinks.length) {
                setTimeout(checkNextBatch, 100);
            }
        }
        checkNextBatch();
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', markLinks);
    } else {
        markLinks();
    }
})();
</script>
<div class="offline-indicator">OFFLINE PREVIEW</div>
'''
            if '</body>' in content.lower():
                content = content.replace('</body>', f'{indicator}</body>')
                content = content.replace('</BODY>', f'{indicator}</BODY>')
            else:
                content = content + indicator

            return Response(content=content, media_type="text/html")

        except Exception as e:
            raise HTTPException(500, f"Error reading file: {str(e)}")

    return FileResponse(file_path, media_type=content_type)


@router.get("/assets/{source_id}/{asset_path:path}")
async def serve_asset(source_id: str, asset_path: str):
    """
    Serve static assets (CSS, JS, images) from source folder.

    For filesystem sources: looks in common asset locations.
    For ZIM sources: falls back to serving from ZIM file.
    """
    import urllib.parse

    asset_path = urllib.parse.unquote(asset_path)

    backup = get_backup_folder()
    if not backup:
        raise HTTPException(500, "Backup folder not configured")

    source_path = backup / source_id

    # Try various filesystem locations first
    possible_paths = [
        source_path / asset_path,
        source_path / "pages" / asset_path,
        source_path / "assets" / asset_path,
        source_path / "static" / asset_path,
    ]

    for file_path in possible_paths:
        try:
            resolved = file_path.resolve()
            if str(resolved).startswith(str(source_path.resolve())) and resolved.exists():
                content_type, _ = mimetypes.guess_type(str(resolved))
                return FileResponse(resolved, media_type=content_type)
        except Exception:
            continue

    # For ZIM sources, try serving from ZIM file
    manifest = get_source_manifest(source_id)
    if manifest and manifest.get("created_from") == "zim_import":
        zim_path = manifest.get("zim_path")
        if zim_path and Path(zim_path).exists():
            result = serve_zim_asset(zim_path, asset_path)
            if result:
                content, mimetype = result
                if not mimetype:
                    mimetype, _ = mimetypes.guess_type(asset_path)
                    if not mimetype:
                        mimetype = "application/octet-stream"
                return Response(content=content, media_type=mimetype)

            # Return empty placeholders for missing CSS/JS
            if asset_path.endswith('.css'):
                return Response(content=b"/* Asset not in ZIM */", media_type="text/css")
            elif asset_path.endswith('.js'):
                return Response(content=b"/* Asset not in ZIM */", media_type="application/javascript")

    raise HTTPException(404, f"Asset not found: {asset_path}")


@router.get("/debug/zim-info/{source_id}")
async def debug_zim_info(source_id: str):
    """Debug endpoint: Get basic info about a ZIM file."""
    manifest = get_source_manifest(source_id)
    if not manifest:
        raise HTTPException(404, f"Source not found: {source_id}")

    if manifest.get("created_from") != "zim_import":
        return {"error": "Not a ZIM source"}

    zim_path = manifest.get("zim_path")
    if not zim_path or not Path(zim_path).exists():
        raise HTTPException(404, "ZIM file not found")

    zim = get_cached_zim(zim_path)
    if not zim:
        raise HTTPException(500, "Failed to open ZIM file")

    info = {
        "source_id": source_id,
        "zim_path": zim_path,
        "zim_exists": Path(zim_path).exists(),
        "zim_size_mb": round(Path(zim_path).stat().st_size / 1024 / 1024, 2),
    }

    # Try to get ZIM metadata using header_fields (correct zimply_core API)
    try:
        header = getattr(zim, 'header_fields', {})
        article_count = header.get('articleCount', 0)
        info["article_count"] = article_count
        info["header_fields"] = header

        # Sample URLs by iterating through article indices (correct way)
        sample_urls = []
        sample_images = []
        sample_assets = []

        for i in range(min(article_count, 500)):  # Check first 500 entries
            try:
                article = zim.get_article_by_id(i)
                if article is None:
                    continue

                url = getattr(article, 'url', '') or ''
                namespace = getattr(article, 'namespace', '')
                mimetype = str(getattr(article, 'mimetype', ''))

                entry = {
                    "index": i,
                    "url": url,
                    "namespace": namespace,
                    "mimetype": mimetype
                }

                # Categorize entries
                if namespace == 'I' or 'image' in mimetype.lower():
                    if len(sample_images) < 10:
                        sample_images.append(entry)
                elif namespace == '-' or url.startswith('-/'):
                    if len(sample_assets) < 10:
                        sample_assets.append(entry)
                elif namespace == 'A' or 'html' in mimetype.lower():
                    if len(sample_urls) < 10:
                        sample_urls.append(entry)

                # Stop early if we have enough samples
                if len(sample_urls) >= 10 and len(sample_images) >= 10 and len(sample_assets) >= 10:
                    break
            except Exception:
                continue

        info["sample_articles"] = sample_urls
        info["sample_images"] = sample_images
        info["sample_assets"] = sample_assets
        info["samples_checked"] = min(article_count, 500)
    except Exception as e:
        info["error"] = str(e)
        import traceback
        info["traceback"] = traceback.format_exc()

    return info


@router.get("/debug/zim-search/{source_id}")
async def debug_zim_search(source_id: str, pattern: str = ""):
    """
    Debug endpoint: Search for entries in a ZIM file matching a pattern.
    Use this to discover how images/assets are stored in the ZIM.

    Args:
        source_id: Source to search
        pattern: Substring to search for in ZIM URLs (e.g., "Elvis" or ".jpg")
    """
    manifest = get_source_manifest(source_id)
    if not manifest:
        raise HTTPException(404, f"Source not found: {source_id}")

    if manifest.get("created_from") != "zim_import":
        return {"error": "Not a ZIM source", "source_type": manifest.get("created_from")}

    zim_path = manifest.get("zim_path")
    if not zim_path or not Path(zim_path).exists():
        raise HTTPException(404, "ZIM file not found")

    zim = get_cached_zim(zim_path)
    if not zim:
        raise HTTPException(500, "Failed to open ZIM file")

    matches = []
    try:
        # Get article count from header (correct zimply_core API)
        header = getattr(zim, 'header_fields', {})
        article_count = header.get('articleCount', 0)

        # Iterate through article indices
        for i in range(min(article_count, 10000)):  # Limit to prevent timeout
            try:
                article = zim.get_article_by_id(i)
                if article is None:
                    continue

                url = getattr(article, 'url', '') or ''
                if pattern.lower() in url.lower():
                    namespace = getattr(article, 'namespace', '')
                    mimetype = str(getattr(article, 'mimetype', ''))
                    matches.append({
                        "index": i,
                        "url": url,
                        "namespace": namespace,
                        "mimetype": mimetype
                    })
                    if len(matches) >= 50:  # Limit results
                        break
            except Exception:
                continue
    except Exception as e:
        import traceback
        return {"error": f"Search failed: {str(e)}", "traceback": traceback.format_exc(), "partial_matches": matches}

    return {
        "source_id": source_id,
        "pattern": pattern,
        "match_count": len(matches),
        "matches": matches,
        "total_entries": article_count if 'article_count' in dir() else "unknown",
        "note": "Use these URLs directly in generate_zim_url_variations to add support"
    }
