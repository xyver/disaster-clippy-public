"""
HTML Backup Scraper

Downloads entire websites to local HTML files for offline access.
Supports MediaWiki and static sites.

Uses backup_manifest.json for backup tracking.
"""

import os
import json
import time
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse, unquote
from bs4 import BeautifulSoup

from ..schemas import get_backup_manifest_file, CURRENT_SCHEMA_VERSION


class HTMLBackupScraper:
    """
    Scrapes websites and saves raw HTML + assets for offline viewing.
    """

    def __init__(self, backup_path, source_id, base_url, scraper_type="mediawiki", sitemap_url=None):
        self.backup_path = Path(backup_path)
        self.source_id = source_id
        self.base_url = base_url.rstrip("/")
        self.scraper_type = scraper_type
        self.sitemap_url = sitemap_url  # Optional custom sitemap URL

        # Create output directories (flat structure - source_id folder directly in backup root)
        self.output_dir = self.backup_path / source_id
        self.pages_dir = self.output_dir / "pages"
        self.assets_dir = self.output_dir / "assets"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(exist_ok=True)
        self.assets_dir.mkdir(exist_ok=True)

        # Track progress
        self.visited_urls = set()
        self.downloaded_assets = set()
        self.pages_saved = 0
        self.errors = []

        # Request settings
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Load existing manifest if resuming
        # V3 naming convention: backup_manifest.json
        self.manifest_path = self.output_dir / get_backup_manifest_file()
        # Legacy paths for migration
        self.legacy_manifest_paths = [
            self.output_dir / f"{source_id}_backup_manifest.json",  # v2 format
            self.output_dir / "manifest.json",  # very old format
        ]
        self.manifest = self._load_manifest()

    def _load_manifest(self):
        """Load existing manifest or create new one. Migrates from legacy formats if needed."""
        # Try v3 format first
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Check for legacy formats and migrate
        for legacy_path in self.legacy_manifest_paths:
            if legacy_path.exists():
                print(f"Migrating {legacy_path.name} to {self.manifest_path.name}")
                with open(legacy_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                # Update schema version
                manifest["schema_version"] = CURRENT_SCHEMA_VERSION
                # Save to new location
                with open(self.manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)
                # Remove legacy file
                legacy_path.unlink()
                return manifest

        # Create new manifest
        return {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "source_id": self.source_id,
            "base_url": self.base_url,
            "scraper_type": self.scraper_type,
            "created_at": datetime.now().isoformat(),
            "last_updated": None,
            "pages": {},
            "assets": {},
            "total_pages": 0,
            "total_size_bytes": 0
        }

    def _save_manifest(self):
        """Save manifest to disk"""
        self.manifest["last_updated"] = datetime.now().isoformat()
        self.manifest["total_pages"] = len(self.manifest["pages"])
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    def _url_to_filename(self, url):
        """Convert URL to safe filename"""
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            path = "index"
        # Handle wiki paths
        if "/wiki/" in path:
            path = path.split("/wiki/")[-1]
        # Make safe filename
        safe_name = path.replace("/", "_").replace(":", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._-")
        if len(safe_name) > 100:
            safe_name = safe_name[:100] + "_" + hashlib.md5(url.encode()).hexdigest()[:8]
        return safe_name + ".html"

    def _fetch_page(self, url):
        """
        Fetch a page with error handling.

        Returns tuple: (html_content, error_type)
        - (content, None) on success
        - (None, "404") on 404 Not Found (dead link - expected)
        - (None, "server") on 5xx, timeout, connection error (server issue)
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text, None
        except requests.exceptions.HTTPError as e:
            error_msg = f"Failed to fetch {url}: {str(e)}"
            self.errors.append(error_msg)
            print(f"  [ERROR] {error_msg}")
            # 404 = dead link (expected), 5xx = server error
            if response.status_code == 404:
                return None, "404"
            elif response.status_code >= 500:
                return None, "server"
            else:
                # 403, 401, etc. - treat as dead link, not server issue
                return None, "client"
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            error_msg = f"Failed to fetch {url}: {str(e)}"
            self.errors.append(error_msg)
            print(f"  [ERROR] {error_msg}")
            return None, "server"
        except Exception as e:
            error_msg = f"Failed to fetch {url}: {str(e)}"
            self.errors.append(error_msg)
            print(f"  [ERROR] {error_msg}")
            return None, "unknown"

    def _get_wiki_pages(self, limit=100):
        """Get list of wiki pages using MediaWiki API"""
        pages = []

        # Try Fandom/Wikia API first
        if "fandom.com" in self.base_url or "wikia.com" in self.base_url:
            api_url = f"{self.base_url}/api.php"
        else:
            api_url = f"{self.base_url}/w/api.php"

        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": min(limit, 500),
            "format": "json"
        }

        try:
            response = self.session.get(api_url, params=params, timeout=30)
            data = response.json()

            if "query" in data and "allpages" in data["query"]:
                for page in data["query"]["allpages"]:
                    title = page["title"]
                    # Build wiki URL
                    page_url = f"{self.base_url}/wiki/{title.replace(' ', '_')}"
                    pages.append({"title": title, "url": page_url})

                    if len(pages) >= limit:
                        break

        except Exception as e:
            self.errors.append(f"API error: {str(e)}")
            # Fallback: try to scrape from Special:AllPages
            self._get_wiki_pages_fallback(pages, limit)

        return pages[:limit]

    def _get_wiki_pages_fallback(self, pages, limit):
        """Fallback method to get wiki pages by scraping"""
        try:
            all_pages_url = f"{self.base_url}/wiki/Special:AllPages"
            html, _ = self._fetch_page(all_pages_url)
            if html:
                soup = BeautifulSoup(html, "html.parser")
                for link in soup.select(".mw-allpages-body a, .allpageslist a"):
                    href = link.get("href", "")
                    if "/wiki/" in href and ":" not in href.split("/wiki/")[-1]:
                        title = unquote(href.split("/wiki/")[-1].replace("_", " "))
                        page_url = urljoin(self.base_url, href)
                        pages.append({"title": title, "url": page_url})
                        if len(pages) >= limit:
                            break
        except Exception as e:
            self.errors.append(f"Fallback scrape error: {str(e)}")

    def _get_sitemap_pages(self, limit=100, sitemap_path="/sitemap.xml"):
        """Get list of pages from sitemap.xml for static sites"""
        pages = []

        # Use custom sitemap URL if provided, otherwise construct from base_url
        if self.sitemap_url:
            sitemap_url = self.sitemap_url
        else:
            sitemap_url = f"{self.base_url}{sitemap_path}"

        try:
            print(f"Fetching sitemap from {sitemap_url}")
            response = self.session.get(sitemap_url, timeout=30)
            response.raise_for_status()

            # Try xml parser first
            soup = BeautifulSoup(response.text, "xml")

            # Check for sitemap index (multiple sitemaps)
            sitemap_tags = soup.find_all("sitemap")
            if sitemap_tags:
                print(f"Found sitemap index with {len(sitemap_tags)} sub-sitemaps")
                for sitemap in sitemap_tags:
                    loc = sitemap.find("loc")
                    if loc and len(pages) < limit:
                        sub_pages = self._fetch_sub_sitemap(loc.text, limit - len(pages))
                        pages.extend(sub_pages)
            else:
                # Regular sitemap - extract URLs
                url_tags = soup.find_all("url")

                # If xml parser found nothing, try html.parser
                if len(url_tags) == 0:
                    print("XML parser found 0 URLs, trying html.parser...")
                    soup = BeautifulSoup(response.text, "html.parser")
                    url_tags = soup.find_all("url")

                # If still nothing, use regex fallback
                if len(url_tags) == 0:
                    print("BeautifulSoup found 0 URLs, using regex fallback...")
                    import re
                    loc_pattern = re.compile(r'<loc[^>]*>([^<]+)</loc>', re.IGNORECASE)
                    loc_matches = loc_pattern.findall(response.text)
                    print(f"Regex found {len(loc_matches)} <loc> tags")

                    if len(loc_matches) > 0:
                        # Found XML-style loc tags via regex
                        for url in loc_matches:
                            url = url.strip()
                            lower_url = url.lower()
                            if any(lower_url.endswith(ext) for ext in [
                                '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip',
                                '.doc', '.docx', '.xls', '.xlsx', '.mp3', '.mp4'
                            ]):
                                continue

                            parsed = urlparse(url)
                            path = parsed.path.strip("/")
                            title = path.split("/")[-1] if path else "Home"
                            title = title.replace(".htm", "").replace(".html", "").replace("_", " ").replace("-", " ")
                            if not title:
                                title = "Home"

                            pages.append({"title": title, "url": url})
                            if len(pages) >= limit:
                                break
                    else:
                        # No XML sitemap - check if it's an HTML page with links
                        content_type = response.headers.get("Content-Type", "").lower()
                        if "text/html" in content_type or response.text.strip().startswith("<!"):
                            print("Detected HTML page, extracting links...")
                            html_soup = BeautifulSoup(response.text, "html.parser")
                            seen_urls = set()

                            for link in html_soup.find_all("a", href=True):
                                href = link["href"]
                                full_url = urljoin(sitemap_url, href)

                                # Only include internal links
                                if not full_url.startswith(self.base_url):
                                    continue

                                # Skip anchors
                                if "#" in full_url:
                                    full_url = full_url.split("#")[0]

                                lower_url = full_url.lower()
                                if any(lower_url.endswith(ext) for ext in [
                                    '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip',
                                    '.doc', '.docx', '.xls', '.xlsx', '.mp3', '.mp4', '.css', '.js'
                                ]):
                                    continue

                                if full_url not in seen_urls:
                                    seen_urls.add(full_url)
                                    # Get title from link text or URL
                                    link_text = link.get_text(strip=True)
                                    if link_text and len(link_text) < 100:
                                        title = link_text
                                    else:
                                        parsed = urlparse(full_url)
                                        path = parsed.path.strip("/")
                                        title = path.split("/")[-1] if path else "Home"
                                        title = title.replace(".htm", "").replace(".html", "").replace("_", " ")

                                    pages.append({"title": title, "url": full_url})
                                    if len(pages) >= limit:
                                        break

                            print(f"Extracted {len(pages)} links from HTML page")
                else:
                    # Process url_tags from BeautifulSoup
                    for url_tag in url_tags:
                        loc = url_tag.find("loc")
                        if loc:
                            url = loc.text
                            # Skip non-HTML content
                            lower_url = url.lower()
                            if any(lower_url.endswith(ext) for ext in [
                                '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip',
                                '.doc', '.docx', '.xls', '.xlsx', '.mp3', '.mp4'
                            ]):
                                continue

                            # Extract title from URL path
                            parsed = urlparse(url)
                            path = parsed.path.strip("/")
                            title = path.split("/")[-1] if path else "Home"
                            title = title.replace(".htm", "").replace(".html", "").replace("_", " ").replace("-", " ")
                            if not title:
                                title = "Home"

                            pages.append({"title": title, "url": url})

                            if len(pages) >= limit:
                                break

            print(f"Found {len(pages)} pages in sitemap")

        except Exception as e:
            error_msg = f"Sitemap error: {str(e)}"
            self.errors.append(error_msg)
            print(f"  [ERROR] {error_msg}")
            # Fallback: crawl from homepage
            print("Falling back to link crawling...")
            pages = self._crawl_links(limit)

        return pages[:limit]

    def _fetch_sub_sitemap(self, sitemap_url, limit):
        """Fetch URLs from a sub-sitemap"""
        pages = []
        try:
            response = self.session.get(sitemap_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "xml")
            for url_tag in soup.find_all("url"):
                loc = url_tag.find("loc")
                if loc:
                    url = loc.text
                    lower_url = url.lower()
                    if any(lower_url.endswith(ext) for ext in [
                        '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip'
                    ]):
                        continue

                    parsed = urlparse(url)
                    path = parsed.path.strip("/")
                    title = path.split("/")[-1] if path else "Home"
                    title = title.replace(".htm", "").replace(".html", "").replace("_", " ")

                    pages.append({"title": title, "url": url})
                    if len(pages) >= limit:
                        break

        except Exception as e:
            self.errors.append(f"Sub-sitemap error {sitemap_url}: {str(e)}")

        return pages

    def _crawl_links(self, limit=100):
        """Crawl site by following links (fallback if no sitemap)"""
        pages = []
        visited = set()
        to_visit = [self.base_url]

        while to_visit and len(pages) < limit:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)

            html, _ = self._fetch_page(url)
            if not html:
                continue

            # Extract title
            soup = BeautifulSoup(html, "html.parser")
            title_elem = soup.find("title")
            title = title_elem.get_text(strip=True) if title_elem else url.split("/")[-1]

            pages.append({"title": title, "url": url})

            # Find links to follow
            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(url, href)

                # Only follow internal links
                if not full_url.startswith(self.base_url):
                    continue

                # Skip anchors
                if "#" in full_url:
                    full_url = full_url.split("#")[0]

                # Skip non-HTML
                lower_url = full_url.lower()
                if any(lower_url.endswith(ext) for ext in ['.jpg', '.png', '.pdf', '.zip']):
                    continue

                if full_url not in visited and full_url not in to_visit:
                    to_visit.append(full_url)

            time.sleep(0.3)  # Be nice

        return pages

    def _download_asset(self, asset_url, include_assets=True):
        """Download an asset (image, CSS, JS) if not already downloaded"""
        if not include_assets:
            return None

        if asset_url in self.downloaded_assets:
            return self.manifest["assets"].get(asset_url, {}).get("local_path")

        try:
            # Parse and normalize URL
            if asset_url.startswith("//"):
                asset_url = "https:" + asset_url
            elif asset_url.startswith("/"):
                asset_url = urljoin(self.base_url, asset_url)

            # Skip external assets
            if urlparse(asset_url).netloc not in urlparse(self.base_url).netloc:
                return None

            response = self.session.get(asset_url, timeout=30)
            response.raise_for_status()

            # Determine filename
            parsed = urlparse(asset_url)
            filename = os.path.basename(parsed.path)
            if not filename or len(filename) > 100:
                ext = os.path.splitext(parsed.path)[1] or ".bin"
                filename = hashlib.md5(asset_url.encode()).hexdigest()[:16] + ext

            local_path = self.assets_dir / filename
            with open(local_path, 'wb') as f:
                f.write(response.content)

            self.downloaded_assets.add(asset_url)
            self.manifest["assets"][asset_url] = {
                "local_path": f"assets/{filename}",
                "size": len(response.content)
            }

            return f"assets/{filename}"

        except Exception as e:
            # Don't log every asset error, can be noisy
            return None

    def _process_html(self, html, page_url, include_assets=True):
        """Process HTML to fix links and optionally download assets"""
        soup = BeautifulSoup(html, "html.parser")

        # Remove scripts, tracking, and ads
        for tag in soup.find_all(["script", "iframe", "noscript"]):
            tag.decompose()

        # Remove ad-related elements by exact class/id names (not substrings!)
        # Using exact match or prefix/suffix match to avoid false positives
        # e.g. "ad" should not match "header" or "breadcrumb"
        ad_exact_classes = [
            # Exact class names for ads
            "ad", "ads", "advert", "advertisement",
            "ad-container", "ad-wrapper", "ad-slot", "ad-unit",
            "banner-ad", "sidebar-ad", "leaderboard", "skyscraper",
            # Fandom/Wikia specific
            "global-navigation", "fandom-sticky-header", "page-side-tools",
            "top-ads-container", "bottom-ads-container", "rail-module",
            "wikia-ad", "featured-video", "mcf-wrapper", "notifications-placeholder",
            "fandom-community-header__background",
            # General tracking/social
            "social-share", "share-buttons", "cookie-notice", "cookie-banner",
            "newsletter-signup", "modal-backdrop",
            # Analytics
            "google-analytics", "tracking-pixel", "beacon"
        ]

        def class_matches_ad(class_list):
            """Check if any class in the list is an ad-related class"""
            if not class_list:
                return False
            for cls in class_list:
                cls_lower = cls.lower()
                # Check exact match
                if cls_lower in ad_exact_classes:
                    return True
                # Check if class starts with ad- or ends with -ad
                if cls_lower.startswith('ad-') or cls_lower.startswith('ads-'):
                    return True
                if cls_lower.endswith('-ad') or cls_lower.endswith('-ads'):
                    return True
            return False

        for elem in soup.find_all(class_=class_matches_ad):
            elem.decompose()

        # Remove by exact id match
        for elem in soup.find_all(id=lambda x: x and x.lower() in ad_exact_classes if x else False):
            elem.decompose()

        # Remove elements with ad-related data attributes
        for elem in soup.find_all(attrs={"data-ad": True}):
            elem.decompose()
        for elem in soup.find_all(attrs={"data-advertisement": True}):
            elem.decompose()

        # Remove inline ad scripts and tracking pixels
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if any(x in src.lower() for x in ["doubleclick", "googlesyndication", "adservice", "tracking", "pixel", "beacon", "analytics"]):
                img.decompose()

        # Process images
        if include_assets:
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src")
                if src:
                    local_path = self._download_asset(src)
                    if local_path:
                        img["src"] = f"../{local_path}"

        # Process CSS links
        if include_assets:
            for link in soup.find_all("link", rel="stylesheet"):
                href = link.get("href")
                if href:
                    local_path = self._download_asset(href)
                    if local_path:
                        link["href"] = f"../{local_path}"

        # Convert internal wiki links to local files
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/wiki/" in href and not href.startswith("http"):
                # Convert to local file reference
                page_name = href.split("/wiki/")[-1]
                local_file = self._url_to_filename(f"{self.base_url}/wiki/{page_name}")
                a["href"] = local_file

        return str(soup)

    def _check_file_valid(self, filepath, min_size=100):
        """Check if a backed up HTML file is valid (not empty/corrupted)"""
        if not filepath.exists():
            return False
        size = filepath.stat().st_size
        return size >= min_size

    def get_corrupted_pages(self, min_size=100):
        """
        Find pages in manifest whose HTML files are empty or corrupted.

        Args:
            min_size: Minimum file size in bytes to be considered valid (default 100)

        Returns:
            List of URLs that need to be re-downloaded
        """
        corrupted = []
        pages = self.manifest.get("pages", {})

        for url, info in pages.items():
            filename = info.get("filename")
            if not filename:
                corrupted.append(url)
                continue

            filepath = self.pages_dir / filename
            if not self._check_file_valid(filepath, min_size):
                corrupted.append(url)

        return corrupted

    def repair(self, progress_callback=None, min_size=100):
        """
        Re-download pages that have empty or corrupted HTML files.

        Args:
            progress_callback: Function(current, total, message) for progress updates
            min_size: Minimum file size to be considered valid

        Returns:
            Dict with repair results
        """
        corrupted = self.get_corrupted_pages(min_size)

        if not corrupted:
            return {
                "success": True,
                "repaired": 0,
                "message": "No corrupted files found"
            }

        print(f"Found {len(corrupted)} corrupted/empty files to repair")

        repaired = 0
        errors = []

        for i, url in enumerate(corrupted):
            page_info = self.manifest["pages"].get(url, {})
            title = page_info.get("title", url.split("/")[-1])

            if progress_callback:
                progress_callback(i + 1, len(corrupted), f"Repairing: {title}")

            print(f"[{i+1}/{len(corrupted)}] Repairing: {title}")

            html, _ = self._fetch_page(url)
            if not html:
                errors.append(f"Failed to fetch: {url}")
                continue

            # Process HTML
            processed_html = self._process_html(html, url, include_assets=False)

            # Save to file
            filename = page_info.get("filename") or self._url_to_filename(url)
            filepath = self.pages_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(processed_html)

            # Update manifest
            self.manifest["pages"][url] = {
                "title": title,
                "filename": filename,
                "saved_at": datetime.now().isoformat(),
                "repaired": True
            }

            repaired += 1
            time.sleep(0.5)  # Be nice to servers

            # Save manifest periodically
            if i % 10 == 0:
                self._save_manifest()

        self._save_manifest()

        return {
            "success": True,
            "repaired": repaired,
            "total_corrupted": len(corrupted),
            "errors": errors
        }

    def backup(self, page_limit=100, include_assets=True, progress_callback=None, priority_urls=None,
               max_consecutive_failures=5, stop_callback=None, follow_links=True, max_depth=3,
               request_delay=0.5):
        """
        Main backup method.

        Args:
            page_limit: Maximum pages to download
            include_assets: Whether to download images/CSS
            progress_callback: Function(current, total, message) for progress updates
            priority_urls: List of URLs to prioritize (e.g., already indexed pages)
            max_consecutive_failures: Stop after this many consecutive failures (default 5)
            stop_callback: Function() that returns True if backup should stop
            follow_links: Whether to discover and follow links from downloaded pages (default True)
            max_depth: Maximum link depth to follow (default 3, 0=only initial pages)
            request_delay: Seconds to wait between requests (default 0.5)
        """
        print(f"Starting HTML backup of {self.source_id}")
        print(f"Output directory: {self.output_dir}")
        print(f"Follow links: {follow_links}, Max depth: {max_depth}, Delay: {request_delay}s")

        # Track URL depths for link following
        url_depths = {}  # url -> depth level

        # Get list of pages
        if progress_callback:
            progress_callback(0, page_limit, "Getting page list...")

        if self.scraper_type == "mediawiki":
            pages = self._get_wiki_pages(limit=page_limit)
        else:
            # For static sites, use sitemap (with crawl fallback)
            pages = self._get_sitemap_pages(limit=page_limit)

        # Mark initial pages as depth 0
        for p in pages:
            url_depths[p["url"]] = 0

        if not pages:
            self.errors.append("No pages found to backup")
            return {
                "success": False,
                "pages_saved": 0,
                "errors": self.errors
            }

        # Add priority URLs that might not be in the API results
        # (pages we've already indexed should be backed up first)
        if priority_urls:
            existing_urls = {p["url"] for p in pages}
            for url in priority_urls:
                if url not in existing_urls:
                    # Extract title from URL
                    title = url.split("/wiki/")[-1].replace("_", " ") if "/wiki/" in url else url
                    pages.append({"title": unquote(title), "url": url, "priority": True})

            # Sort pages: priority first, then others
            priority_set = set(priority_urls)
            pages.sort(key=lambda p: (0 if p["url"] in priority_set else 1, p.get("title", "")))
            print(f"Priority URLs from index: {len(priority_urls)}")

        # Enforce page_limit on total pages
        if len(pages) > page_limit:
            print(f"Trimming from {len(pages)} to {page_limit} pages (page_limit)")
            pages = pages[:page_limit]

        # Check how many are already backed up
        manifest_pages = self.manifest.get("pages", {})
        already_backed_up = sum(1 for p in pages if p["url"] in manifest_pages)
        new_pages = [p for p in pages if p["url"] not in manifest_pages]

        # Count how many priority pages need backup
        priority_to_backup = sum(1 for p in new_pages if p.get("url") in (priority_urls or []))

        print(f"Found {len(pages)} pages from sitemap/start page")
        print(f"Existing manifest has {len(manifest_pages)} pages backed up")
        print(f"Of starting pages, {already_backed_up} already in manifest, {len(new_pages)} new")
        if priority_urls:
            print(f"Priority (indexed) pages to backup: {priority_to_backup}")

        # Use a queue for dynamic link discovery
        from collections import deque
        page_queue = deque(new_pages)
        queued_urls = {p["url"] for p in new_pages}  # Track what's already in queue

        # Load saved queue from previous scrape (resume support)
        saved_queue = self.manifest.get("queued_urls", [])
        if saved_queue:
            loaded_count = 0
            for item in saved_queue:
                url = item.get("url") if isinstance(item, dict) else item
                if url and url not in manifest_pages and url not in queued_urls:
                    title = item.get("title", url.split("/")[-1]) if isinstance(item, dict) else url.split("/")[-1]
                    page_queue.append({"title": title, "url": url})
                    queued_urls.add(url)
                    url_depths[url] = 1  # Assume depth 1 for resumed items
                    loaded_count += 1
            if loaded_count > 0:
                print(f"Loaded {loaded_count} URLs from previous queue (resume)")
            # Clear saved queue since we've loaded it
            self.manifest["queued_urls"] = []
            self.manifest["queued_count"] = 0

        # If no new pages but we have backed up pages, re-scan them for link discovery
        if len(page_queue) == 0 and follow_links and manifest_pages:
            print(f"No new pages from sitemap - scanning {min(100, len(manifest_pages))} backed up pages for new links...")
            rescan_count = 0
            for url, info in list(manifest_pages.items())[:100]:
                filename = info.get("filename")
                if not filename:
                    continue
                filepath = self.pages_dir / filename
                if not filepath.exists():
                    continue

                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        html = f.read()
                    soup = BeautifulSoup(html, "html.parser")

                    # Extract links from this page
                    for link in soup.find_all("a", href=True):
                        href = link["href"]
                        full_url = urljoin(url, href)

                        # Only internal links
                        if not full_url.startswith(self.base_url):
                            continue
                        full_url = full_url.split("#")[0].split("?")[0]
                        if not full_url or full_url == url:
                            continue

                        # Skip non-HTML
                        lower_url = full_url.lower()
                        if any(lower_url.endswith(ext) for ext in [
                            '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip',
                            '.doc', '.docx', '.xls', '.xlsx', '.mp3', '.mp4', '.css', '.js'
                        ]):
                            continue

                        # Add to queue if not already backed up or queued
                        if full_url not in manifest_pages and full_url not in queued_urls:
                            link_text = link.get_text(strip=True)
                            link_title = link_text if link_text and len(link_text) < 100 else full_url.split("/")[-1]
                            page_queue.append({"title": link_title, "url": full_url})
                            queued_urls.add(full_url)
                            url_depths[full_url] = 1
                            rescan_count += 1
                except Exception as e:
                    continue

            if rescan_count > 0:
                print(f"    -> Found {rescan_count} new URLs from local page scan")

        # Download pages from queue (with dynamic link discovery)
        consecutive_failures = 0
        stopped_early = False
        pages_processed = 0

        while page_queue and self.pages_saved < page_limit:
            # Check if we should stop
            if stop_callback and stop_callback():
                print("Backup stopped by user")
                stopped_early = True
                break

            page = page_queue.popleft()
            url = page["url"]
            title = page["title"]
            current_depth = url_depths.get(url, 0)

            if url in self.visited_urls:
                continue

            pages_processed += 1

            if progress_callback:
                progress_callback(self.pages_saved + 1, page_limit,
                    f"Downloading: {title} (depth {current_depth}, {len(page_queue)} queued)")

            print(f"[{self.pages_saved + 1}/{page_limit}] (d{current_depth}) {title}")

            html, error_type = self._fetch_page(url)
            if not html:
                # Only count server errors (5xx, timeout) as consecutive failures
                # 404s are expected dead links, not a sign the server is down
                if error_type == "server":
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        error_msg = f"Stopping after {max_consecutive_failures} consecutive server errors. Last errors: {self.errors[-3:]}"
                        print(f"\n{error_msg}")
                        self.errors.append(error_msg)
                        stopped_early = True
                        break
                else:
                    # 404, 403, etc. - dead link, reset consecutive failures
                    consecutive_failures = 0
                continue

            # Reset consecutive failures on success
            consecutive_failures = 0

            # Process HTML
            processed_html = self._process_html(html, url, include_assets)

            # Save to file
            filename = self._url_to_filename(url)
            filepath = self.pages_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(processed_html)

            self.visited_urls.add(url)
            self.pages_saved += 1

            # Update manifest
            self.manifest["pages"][url] = {
                "title": title,
                "filename": filename,
                "saved_at": datetime.now().isoformat(),
                "depth": current_depth
            }

            # Discover and queue new links if follow_links is enabled
            if follow_links and current_depth < max_depth and self.pages_saved < page_limit:
                soup = BeautifulSoup(html, "html.parser")
                links_added = 0

                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    full_url = urljoin(url, href)

                    # Only follow internal links
                    if not full_url.startswith(self.base_url):
                        continue

                    # Clean URL
                    if "#" in full_url:
                        full_url = full_url.split("#")[0]
                    if "?" in full_url:
                        full_url = full_url.split("?")[0]

                    # Skip non-HTML
                    lower_url = full_url.lower()
                    if any(lower_url.endswith(ext) for ext in [
                        '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip',
                        '.doc', '.docx', '.xls', '.xlsx', '.mp3', '.mp4', '.css', '.js'
                    ]):
                        continue

                    # Add to queue if not seen
                    if full_url not in self.visited_urls and full_url not in queued_urls:
                        if full_url not in self.manifest.get("pages", {}):
                            link_text = link.get_text(strip=True)
                            link_title = link_text if link_text and len(link_text) < 100 else full_url.split("/")[-1]

                            page_queue.append({"title": link_title, "url": full_url})
                            queued_urls.add(full_url)
                            url_depths[full_url] = current_depth + 1
                            links_added += 1

                if links_added > 0:
                    print(f"    -> Discovered {links_added} new links (queue: {len(page_queue)})")

            # Be nice to servers
            if request_delay > 0:
                time.sleep(request_delay)

            # Save manifest periodically
            if pages_processed % 10 == 0:
                self._save_manifest()

        # Create index.html
        self._create_index()

        # Save remaining queue URLs for future resume
        remaining_queue = [{"url": p["url"], "title": p["title"]} for p in page_queue]
        self.manifest["queued_urls"] = remaining_queue
        self.manifest["queued_count"] = len(remaining_queue)

        # Final save
        self._save_manifest()

        total_backed_up = len(self.manifest.get("pages", {}))

        print(f"\nBackup complete!")
        print(f"New pages saved: {self.pages_saved}")
        print(f"Total pages backed up: {total_backed_up}")
        print(f"Assets downloaded: {len(self.downloaded_assets)}")
        print(f"Errors: {len(self.errors)}")
        print(f"URLs still in queue: {len(page_queue)}")

        # Show summary of errors if many failures
        if self.errors:
            print(f"\n--- Error Summary ---")
            # Show first 5 unique error types
            error_types = {}
            for err in self.errors:
                # Extract error type (e.g., "403", "404", "timeout")
                if "403" in err:
                    error_types["403 Forbidden"] = error_types.get("403 Forbidden", 0) + 1
                elif "404" in err:
                    error_types["404 Dead links"] = error_types.get("404 Dead links", 0) + 1
                elif "timeout" in err.lower() or "timed out" in err.lower():
                    error_types["Timeout"] = error_types.get("Timeout", 0) + 1
                elif "connection" in err.lower():
                    error_types["Connection Error"] = error_types.get("Connection Error", 0) + 1
                elif "SSL" in err or "certificate" in err.lower():
                    error_types["SSL/Certificate Error"] = error_types.get("SSL/Certificate Error", 0) + 1
                else:
                    error_types["Other"] = error_types.get("Other", 0) + 1

            for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
                print(f"  {err_type}: {count}")

            # Show first 3 actual error messages
            print(f"\nFirst few errors:")
            for err in self.errors[:3]:
                print(f"  - {err[:100]}...")

        # Determine success - if we got some pages, it's partial success
        success = self.pages_saved > 0 or total_backed_up > 0
        if len(self.errors) > 0 and self.pages_saved == 0 and total_backed_up == 0:
            success = False

        # Calculate remaining queued URLs
        queued_remaining = len(page_queue)

        return {
            "success": success,
            "pages_saved": self.pages_saved,
            "total_backed_up": total_backed_up,
            "skipped": already_backed_up,
            "assets_downloaded": len(self.downloaded_assets),
            "output_dir": str(self.output_dir),
            "errors": self.errors,
            "error_summary": error_types if self.errors else {},
            "stopped_early": stopped_early,
            "queued_remaining": queued_remaining
        }

    def _create_index(self):
        """Create an index.html for easy browsing"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.source_id} - Offline Backup</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .info {{ background: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
        ul {{ list-style: none; padding: 0; }}
        li {{ padding: 5px 0; border-bottom: 1px solid #eee; }}
        a {{ color: #0066cc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>{self.source_id} - Offline Backup</h1>
    <div class="info">
        <p><strong>Source:</strong> {self.base_url}</p>
        <p><strong>Pages:</strong> {len(self.manifest['pages'])}</p>
        <p><strong>Backed up:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    <h2>Pages</h2>
    <ul>
"""
        for url, info in sorted(self.manifest["pages"].items(), key=lambda x: x[1]["title"]):
            html += f'        <li><a href="pages/{info["filename"]}">{info["title"]}</a></li>\n'

        html += """    </ul>
</body>
</html>
"""
        with open(self.output_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(html)


def run_backup(backup_path, source_id, base_url, scraper_type="mediawiki",
               page_limit=100, include_assets=True, progress_callback=None, priority_urls=None,
               max_consecutive_failures=5, stop_callback=None, sitemap_url=None,
               follow_links=True, max_depth=3, request_delay=0.5):
    """
    Convenience function to run a backup.

    Args:
        backup_path: Base path for backups
        source_id: Unique identifier for this source
        base_url: Website base URL
        scraper_type: 'mediawiki' or 'static'
        page_limit: Max pages to download
        include_assets: Download images/CSS
        progress_callback: Progress update function
        priority_urls: URLs to backup first (e.g., already indexed pages)
        max_consecutive_failures: Stop after this many consecutive failures (default 5)
        stop_callback: Function that returns True to stop backup
        sitemap_url: Optional direct URL to sitemap (if not at /sitemap.xml)
        follow_links: Whether to discover and follow links from downloaded pages (default True)
        max_depth: Maximum link depth to follow (default 3, 0=only initial pages)
        request_delay: Seconds to wait between requests (default 0.5)

    Returns dict with success status, pages saved, errors, etc.
    """
    scraper = HTMLBackupScraper(backup_path, source_id, base_url, scraper_type, sitemap_url)
    return scraper.backup(page_limit, include_assets, progress_callback, priority_urls,
                         max_consecutive_failures, stop_callback, follow_links, max_depth,
                         request_delay)


def repair_backup(backup_path, source_id, base_url, scraper_type="mediawiki",
                  progress_callback=None, min_size=100):
    """
    Repair a backup by re-downloading empty/corrupted files.

    Args:
        backup_path: Base path for backups
        source_id: Unique identifier for this source
        base_url: Website base URL
        scraper_type: 'mediawiki' or 'static'
        progress_callback: Progress update function
        min_size: Minimum file size to be considered valid

    Returns dict with repair results.
    """
    scraper = HTMLBackupScraper(backup_path, source_id, base_url, scraper_type)
    return scraper.repair(progress_callback, min_size)


def get_corrupted_count(backup_path, source_id, base_url, scraper_type="mediawiki", min_size=100):
    """
    Check how many files in a backup are corrupted/empty.

    Returns dict with corrupted_count and corrupted_urls.
    """
    scraper = HTMLBackupScraper(backup_path, source_id, base_url, scraper_type)
    corrupted = scraper.get_corrupted_pages(min_size)
    return {
        "corrupted_count": len(corrupted),
        "corrupted_urls": corrupted
    }


def get_backup_status(backup_path, source_id):
    """
    Get the backup status for a source.

    Returns dict with backed_up_count, backed_up_urls, manifest info.
    """
    backup_dir = Path(backup_path) / source_id

    # Check v3 format first, then legacy
    manifest_candidates = [
        backup_dir / get_backup_manifest_file(),  # v3: backup_manifest.json
        backup_dir / f"{source_id}_backup_manifest.json",  # v2 format
        backup_dir / "manifest.json",  # very old format
    ]

    manifest_path = None
    for candidate in manifest_candidates:
        if candidate.exists():
            manifest_path = candidate
            break

    if not manifest_path:
        return {
            "has_backup": False,
            "backed_up_count": 0,
            "backed_up_urls": [],
            "queued_count": 0,
            "last_updated": None
        }

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    return {
        "has_backup": True,
        "backed_up_count": len(manifest.get("pages", {})),
        "backed_up_urls": list(manifest.get("pages", {}).keys()),
        "queued_count": manifest.get("queued_count", 0),
        "last_updated": manifest.get("last_updated"),
        "output_dir": str(backup_dir)
    }


if __name__ == "__main__":
    # Test with Solar Cooking Wiki
    import sys

    backup_path = os.getenv("BACKUP_PATH", "")
    if not backup_path:
        print("Error: BACKUP_PATH not configured.")
        print("Set BACKUP_PATH in .env or configure via Settings page.")
        sys.exit(1)

    result = run_backup(
        backup_path=backup_path,
        source_id="solarcooking",
        base_url="https://solarcooking.fandom.com",
        scraper_type="mediawiki",
        page_limit=20,  # Start small for testing
        include_assets=True
    )

    print("\n" + "="*50)
    print("Result:", json.dumps(result, indent=2))
