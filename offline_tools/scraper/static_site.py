"""
Generic Static HTML Site Scraper.
Works with any static HTML website that has a sitemap.xml.

Usage:
    # Any static site with sitemap
    scraper = StaticSiteScraper("https://www.builditsolar.com", source_name="builditsolar")

    # Get all pages from sitemap
    urls = scraper.get_page_list(limit=100)

    # Scrape a page
    page = scraper.scrape_page(url)
"""

import re
from typing import List, Dict, Optional, Set
from datetime import datetime
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

from .base import BaseScraper, ScrapedPage


class StaticSiteScraper(BaseScraper):
    """
    Generic scraper for static HTML websites.
    Uses sitemap.xml for page discovery and HTML parsing for content extraction.
    """

    def __init__(self, base_url: str, source_name: Optional[str] = None,
                 sitemap_path: str = "/sitemap.xml", rate_limit: float = 1.0,
                 content_selectors: Optional[List[str]] = None,
                 nav_selectors: Optional[List[str]] = None):
        """
        Args:
            base_url: Base URL of the site (e.g., "https://www.builditsolar.com")
            source_name: Identifier for this source (auto-detected from URL if not provided)
            sitemap_path: Path to sitemap.xml (default "/sitemap.xml")
            rate_limit: Minimum seconds between requests
            content_selectors: CSS selectors for main content (tries in order)
            nav_selectors: CSS selectors for navigation elements to remove
        """
        # Auto-detect source name from domain
        if source_name is None:
            parsed = urlparse(base_url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            source_name = domain.split(".")[0]

        super().__init__(
            source_name=source_name,
            base_url=base_url.rstrip("/"),
            rate_limit=rate_limit
        )

        self.sitemap_url = f"{self.base_url}{sitemap_path}"

        # Default content selectors (common patterns)
        self.content_selectors = content_selectors or [
            "main",
            "article",
            "#content",
            "#main-content",
            ".content",
            ".post-content",
            ".entry-content",
            "body"  # Fallback
        ]

        # Default navigation/unwanted element selectors
        self.nav_selectors = nav_selectors or [
            "nav", "header", "footer", "aside",
            ".navigation", ".nav", ".menu", ".sidebar",
            ".footer", ".header", "#nav", "#navigation",
            "#menu", "#sidebar", "#footer", "#header",
            ".breadcrumb", ".breadcrumbs",
            "script", "style", "noscript"
        ]

    def get_sitemap_urls(self) -> List[str]:
        """Parse sitemap.xml and return all URLs"""
        self._wait_for_rate_limit()

        try:
            response = self.session.get(self.sitemap_url, timeout=30)
            response.raise_for_status()

            # Parse as XML
            soup = BeautifulSoup(response.text, "xml")

            # Check for sitemap index (multiple sitemaps)
            sitemap_tags = soup.find_all("sitemap")
            if sitemap_tags:
                # It's a sitemap index - fetch each sub-sitemap
                urls = []
                for sitemap in sitemap_tags:
                    loc = sitemap.find("loc")
                    if loc:
                        sub_urls = self._fetch_sub_sitemap(loc.text)
                        urls.extend(sub_urls)
                return urls

            # Regular sitemap - extract URLs
            urls = []
            for url_tag in soup.find_all("url"):
                loc = url_tag.find("loc")
                if loc:
                    urls.append(loc.text)

            return urls

        except Exception as e:
            print(f"Error fetching sitemap: {e}")
            return []

    def _fetch_sub_sitemap(self, sitemap_url: str) -> List[str]:
        """Fetch URLs from a sub-sitemap"""
        self._wait_for_rate_limit()

        try:
            response = self.session.get(sitemap_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "xml")
            urls = []
            for url_tag in soup.find_all("url"):
                loc = url_tag.find("loc")
                if loc:
                    urls.append(loc.text)
            return urls
        except Exception as e:
            print(f"Error fetching sub-sitemap {sitemap_url}: {e}")
            return []

    def get_page_list(self, limit: Optional[int] = None) -> List[str]:
        """Get list of page URLs from sitemap"""
        urls = self.get_sitemap_urls()

        # Filter to only content pages (skip images, PDFs, etc.)
        content_urls = []
        for url in urls:
            lower_url = url.lower()
            # Skip non-HTML content
            if any(lower_url.endswith(ext) for ext in [
                '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip',
                '.doc', '.docx', '.xls', '.xlsx', '.mp3', '.mp4'
            ]):
                continue
            content_urls.append(url)

        if limit:
            return content_urls[:limit]
        return content_urls

    def crawl_links(self, start_url: Optional[str] = None,
                    limit: int = 100,
                    follow_pattern: Optional[str] = None) -> List[str]:
        """
        Crawl site by following links (fallback if no sitemap).

        Args:
            start_url: Starting URL (defaults to base_url)
            limit: Maximum pages to discover
            follow_pattern: Regex pattern for URLs to follow
        """
        start = start_url or self.base_url
        visited: Set[str] = set()
        to_visit: List[str] = [start]
        found_urls: List[str] = []

        while to_visit and len(found_urls) < limit:
            url = to_visit.pop(0)

            if url in visited:
                continue
            visited.add(url)

            # Check URL pattern
            if follow_pattern and not re.search(follow_pattern, url):
                continue

            soup = self._fetch_page(url)
            if not soup:
                continue

            found_urls.append(url)

            # Find links to follow
            for link in soup.find_all("a", href=True):
                href = link["href"]

                # Make absolute
                full_url = urljoin(url, href)

                # Only follow internal links
                if not full_url.startswith(self.base_url):
                    continue

                # Skip anchors, queries for same page
                if "#" in full_url:
                    full_url = full_url.split("#")[0]

                if full_url not in visited and full_url not in to_visit:
                    to_visit.append(full_url)

        return found_urls

    def scrape_page(self, url: str) -> Optional[ScrapedPage]:
        """Scrape a single page"""
        soup = self._fetch_page(url)
        if not soup:
            return None

        # Extract title
        title = "Unknown"
        title_elem = soup.find("title")
        if title_elem:
            title = title_elem.get_text(strip=True)
        else:
            # Try h1
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)

        # Find main content using selectors
        content_elem = None
        for selector in self.content_selectors:
            if selector.startswith("#"):
                content_elem = soup.find(id=selector[1:])
            elif selector.startswith("."):
                content_elem = soup.find(class_=selector[1:])
            else:
                content_elem = soup.find(selector)

            if content_elem:
                break

        if not content_elem:
            return None

        # Remove navigation and unwanted elements
        for selector in self.nav_selectors:
            if selector.startswith("#"):
                for elem in content_elem.find_all(id=selector[1:]):
                    elem.decompose()
            elif selector.startswith("."):
                for elem in content_elem.find_all(class_=selector[1:]):
                    elem.decompose()
            else:
                for elem in content_elem.find_all(selector):
                    elem.decompose()

        # Get clean text content
        content = content_elem.get_text(separator="\n", strip=True)

        # Clean up whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)

        # Skip very short pages
        if len(content) < 100:
            return None

        # Try to extract categories/topics from URL path or meta
        categories = self._extract_categories(url, soup)

        # Get last modified if available
        last_modified = None
        meta_modified = soup.find("meta", attrs={"name": "last-modified"})
        if meta_modified:
            last_modified = meta_modified.get("content")

        return ScrapedPage(
            url=url,
            title=title,
            content=content,
            source=self.source_name,
            categories=categories[:10],
            last_modified=last_modified,
            content_hash=self._hash_content(content),
            scraped_at=datetime.utcnow().isoformat()
        )

    def _extract_categories(self, url: str, soup: BeautifulSoup) -> List[str]:
        """Extract categories from URL path or page metadata"""
        categories = []

        # Extract from URL path
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split("/") if p and not p.endswith(".htm") and not p.endswith(".html")]
        categories.extend(path_parts)

        # Look for category meta tags
        for meta_name in ["keywords", "category", "categories"]:
            meta = soup.find("meta", attrs={"name": meta_name})
            if meta and meta.get("content"):
                cats = [c.strip() for c in meta["content"].split(",")]
                categories.extend(cats)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for cat in categories:
            if cat.lower() not in seen:
                seen.add(cat.lower())
                unique.append(cat)

        return unique


# Convenience function
def create_static_scraper(url: str, **kwargs) -> StaticSiteScraper:
    """Create a scraper for any static HTML site with sitemap"""
    return StaticSiteScraper(url, **kwargs)


# Specialized wrapper for BuildItSolar
class BuildItSolarScraper(StaticSiteScraper):
    """Scraper specifically configured for builditsolar.com"""

    def __init__(self, rate_limit: float = 1.0):
        super().__init__(
            base_url="https://www.builditsolar.com",
            source_name="builditsolar",
            rate_limit=rate_limit
        )


# Quick test
if __name__ == "__main__":
    print("Testing Static Site Scraper with BuildItSolar...")
    scraper = BuildItSolarScraper(rate_limit=1.0)

    # Test sitemap parsing
    print("\nGetting pages from sitemap...")
    urls = scraper.get_page_list(limit=10)
    print(f"Found {len(urls)} pages")
    for url in urls[:5]:
        print(f"  - {url}")

    # Test page scraping
    if urls:
        # Find a content page (not index)
        content_urls = [u for u in urls if "Projects/" in u]
        test_url = content_urls[0] if content_urls else urls[0]

        print(f"\nScraping: {test_url}")
        page = scraper.scrape_page(test_url)
        if page:
            print(f"  Title: {page.title}")
            print(f"  Content length: {len(page.content)} chars")
            print(f"  Categories: {page.categories}")
            print(f"  First 300 chars: {page.content[:300]}...")
        else:
            print("  Failed to scrape page")
