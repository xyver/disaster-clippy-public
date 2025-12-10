"""
Generic MediaWiki scraper.
Works with any MediaWiki-based wiki (Wikipedia, Fandom, Appropedia, etc.)

Usage:
    # Any MediaWiki site
    scraper = MediaWikiScraper("https://www.appropedia.org", source_name="appropedia")
    scraper = MediaWikiScraper("https://solarcooking.fandom.com", source_name="solarcooking")

    # Search and scrape
    urls = scraper.search_pages("solar cooker", limit=20)
    pages = [scraper.scrape_page(url) for url in urls]
"""

import re
from typing import List, Dict, Optional
from datetime import datetime
from urllib.parse import quote, urlparse

from .base import BaseScraper, ScrapedPage


class MediaWikiScraper(BaseScraper):
    """
    Generic scraper for any MediaWiki-based wiki.
    Uses the MediaWiki API for reliable content extraction.
    """

    def __init__(self, base_url: str, source_name: Optional[str] = None,
                 api_path: str = "/api.php", rate_limit: float = 1.0):
        """
        Args:
            base_url: Base URL of the wiki (e.g., "https://www.appropedia.org")
            source_name: Identifier for this source (auto-detected from URL if not provided)
            api_path: Path to API endpoint (default "/api.php", some use "/w/api.php")
            rate_limit: Minimum seconds between requests
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

        # Detect API path
        self.api_url = self._detect_api_url(api_path)

    def _detect_api_url(self, api_path: str) -> str:
        """Detect the correct API URL for this wiki"""
        # Common API paths
        paths_to_try = [
            api_path,
            "/api.php",
            "/w/api.php",
            "/mediawiki/api.php",
        ]

        for path in paths_to_try:
            url = f"{self.base_url}{path}"
            try:
                self._wait_for_rate_limit()
                response = self.session.get(
                    url,
                    params={"action": "query", "meta": "siteinfo", "format": "json"},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    if "query" in data:
                        return url
            except:
                continue

        # Default fallback
        return f"{self.base_url}{api_path}"

    def _api_request(self, params: Dict) -> Optional[Dict]:
        """Make a request to the MediaWiki API"""
        self._wait_for_rate_limit()
        params["format"] = "json"

        try:
            response = self.session.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API error: {e}")
            return None

    def get_site_info(self) -> Optional[Dict]:
        """Get wiki site information"""
        data = self._api_request({
            "action": "query",
            "meta": "siteinfo",
            "siprop": "general|statistics"
        })
        if data:
            return data.get("query", {})
        return None

    def get_page_list(self, limit: Optional[int] = None) -> List[str]:
        """Get list of all page URLs (main namespace)"""
        return self.get_all_pages(limit=limit)

    def get_all_pages(self, limit: Optional[int] = None, namespace: int = 0) -> List[str]:
        """
        Get all pages from the wiki.

        Args:
            limit: Maximum number of pages (None = all)
            namespace: MediaWiki namespace (0 = main articles)

        Returns:
            List of page URLs
        """
        urls = []
        continue_token = None

        while True:
            params = {
                "action": "query",
                "list": "allpages",
                "apnamespace": namespace,
                "aplimit": min(limit or 500, 500),
            }

            if continue_token:
                params["apcontinue"] = continue_token

            data = self._api_request(params)
            if not data:
                break

            pages = data.get("query", {}).get("allpages", [])
            for page in pages:
                title = page["title"]
                url = self._title_to_url(title)
                urls.append(url)

                if limit and len(urls) >= limit:
                    return urls

            if "continue" in data:
                continue_token = data["continue"].get("apcontinue")
            else:
                break

        return urls

    def search_pages(self, query: str, limit: int = 50) -> List[str]:
        """
        Search for pages matching a query.

        Args:
            query: Search term
            limit: Maximum results

        Returns:
            List of page URLs matching the query
        """
        urls = []
        offset = 0

        while len(urls) < limit:
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srnamespace": 0,
                "srlimit": min(limit - len(urls), 50),
                "sroffset": offset,
            }

            data = self._api_request(params)
            if not data:
                break

            results = data.get("query", {}).get("search", [])
            if not results:
                break

            for result in results:
                url = self._title_to_url(result["title"])
                if url not in urls:
                    urls.append(url)

            offset += len(results)

            if "continue" not in data:
                break

        print(f"Found {len(urls)} pages matching '{query}'")
        return urls[:limit]

    def get_category_pages(self, category: str, limit: int = 100,
                          include_subcats: bool = False) -> List[str]:
        """
        Get all pages in a specific category.

        Args:
            category: Category name (without "Category:" prefix)
            limit: Maximum pages to return
            include_subcats: Whether to include pages from subcategories

        Returns:
            List of page URLs in the category
        """
        urls = []
        categories_to_process = [category]
        processed_categories = set()

        while categories_to_process and len(urls) < limit:
            current_category = categories_to_process.pop(0)
            if current_category in processed_categories:
                continue
            processed_categories.add(current_category)

            continue_token = None

            while len(urls) < limit:
                params = {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": f"Category:{current_category}",
                    "cmtype": "page|subcat" if include_subcats else "page",
                    "cmlimit": min(limit - len(urls), 500),
                }

                if continue_token:
                    params["cmcontinue"] = continue_token

                data = self._api_request(params)
                if not data:
                    break

                members = data.get("query", {}).get("categorymembers", [])
                for member in members:
                    if member.get("ns") == 14 and include_subcats:  # Namespace 14 = Category
                        subcat = member["title"].replace("Category:", "")
                        if subcat not in processed_categories:
                            categories_to_process.append(subcat)
                    elif member.get("ns") == 0:  # Main namespace
                        url = self._title_to_url(member["title"])
                        if url not in urls:
                            urls.append(url)

                if "continue" in data:
                    continue_token = data["continue"].get("cmcontinue")
                else:
                    break

        print(f"Found {len(urls)} pages in category '{category}'")
        return urls[:limit]

    def list_categories(self, limit: int = 100) -> List[str]:
        """List all categories on the wiki"""
        categories = []
        continue_token = None

        while len(categories) < limit:
            params = {
                "action": "query",
                "list": "allcategories",
                "aclimit": min(limit - len(categories), 500),
            }

            if continue_token:
                params["accontinue"] = continue_token

            data = self._api_request(params)
            if not data:
                break

            cats = data.get("query", {}).get("allcategories", [])
            for cat in cats:
                categories.append(cat["*"])

            if "continue" in data:
                continue_token = data["continue"].get("accontinue")
            else:
                break

        return categories[:limit]

    def scrape_page(self, url: str) -> Optional[ScrapedPage]:
        """Scrape a single page by URL"""
        title = self._url_to_title(url)
        return self.scrape_page_by_title(title)

    def scrape_page_by_title(self, title: str) -> Optional[ScrapedPage]:
        """
        Scrape a page by title using MediaWiki API.
        Uses API for clean text extraction, falls back to HTML parsing if
        the TextExtracts extension is not available.
        """
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|categories|revisions",
            "explaintext": "true",
            "exsectionformat": "plain",
            "cllimit": "10",
            "rvprop": "timestamp",
        }

        data = self._api_request(params)
        if not data:
            return None

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None

        # Get first (only) page
        page_data = list(pages.values())[0]

        # Check if page exists
        if "missing" in page_data:
            return None

        page_title = page_data.get("title", title)
        content = page_data.get("extract", "")

        # Get categories from API
        categories = []
        for cat in page_data.get("categories", []):
            cat_name = cat.get("title", "").replace("Category:", "")
            if cat_name:
                categories.append(cat_name)

        # Get last modified from API
        last_modified = None
        revisions = page_data.get("revisions", [])
        if revisions:
            last_modified = revisions[0].get("timestamp")

        # Build URL
        page_url = self._title_to_url(page_title)

        # If API extract is empty/short, fall back to HTML parsing
        internal_links = []
        if len(content) < 100:
            html_result = self._scrape_page_html(page_url)
            if html_result:
                content = html_result["content"]
                # Use HTML categories if API didn't return any
                if not categories:
                    categories = html_result.get("categories", [])
                internal_links = html_result.get("internal_links", [])
            else:
                return None  # Skip if both methods fail
        else:
            # API gave us content, but we still need links from HTML
            html_result = self._scrape_page_html(page_url)
            if html_result:
                internal_links = html_result.get("internal_links", [])

        # Clean content
        content = self._clean_content(content)

        if len(content) < 100:  # Skip stub articles
            return None

        return ScrapedPage(
            url=page_url,
            title=page_title,
            content=content,
            source=self.source_name,
            categories=categories[:10],
            last_modified=last_modified,
            content_hash=self._hash_content(content),
            scraped_at=datetime.utcnow().isoformat(),
            internal_links=internal_links if internal_links else None
        )

    def _scrape_page_html(self, url: str) -> Optional[Dict]:
        """
        Fallback HTML scraping for wikis without TextExtracts extension.
        Returns dict with 'content' and 'categories' keys.
        """
        soup = self._fetch_page(url)
        if not soup:
            return None

        # Find main content div (common MediaWiki selectors)
        content_div = (
            soup.find("div", class_="mw-parser-output") or
            soup.find("div", id="mw-content-text") or
            soup.find("div", id="bodyContent")
        )

        if not content_div:
            return None

        # Remove unwanted elements
        unwanted_classes = [
            "navbox", "infobox", "sidebar", "toc", "mbox", "ambox",
            "notice", "hatnote", "metadata", "reference", "reflist",
            "noprint", "mw-editsection", "printfooter", "catlinks",
            "portal", "sister", "succession", "footer", "navbox-styles"
        ]

        for unwanted in content_div.find_all(["script", "style", "nav", "aside"]):
            unwanted.decompose()

        for unwanted in content_div.find_all(
            ["div", "span", "table"],
            class_=lambda x: x and any(c in str(x).lower() for c in unwanted_classes)
        ):
            unwanted.decompose()

        # Remove edit links
        for edit_link in content_div.find_all("span", class_="mw-editsection"):
            edit_link.decompose()

        # Remove reference markers [1], [2], etc.
        for ref in content_div.find_all("sup", class_="reference"):
            ref.decompose()

        # Get text content
        content = content_div.get_text(separator="\n", strip=True)

        # Extract categories from page
        categories = []
        cat_links = soup.find_all("a", href=re.compile(r"/wiki/Category:"))
        for link in cat_links:
            cat_name = link.get_text(strip=True)
            if cat_name and cat_name not in categories:
                categories.append(cat_name)

        # Extract internal links
        internal_links = self._extract_internal_links(content_div)

        return {
            "content": content,
            "categories": categories,
            "internal_links": internal_links
        }

    def _extract_internal_links(self, content_elem) -> List[str]:
        """
        Extract internal wiki links from content.
        Returns list of page titles that this page links to.
        """
        if not content_elem:
            return []

        internal_links = []
        seen = set()

        for link in content_elem.find_all("a", href=True):
            href = link["href"]

            # Only include wiki article links
            if "/wiki/" not in href:
                continue

            # Skip special pages, files, categories
            if any(x in href for x in ["/wiki/Special:", "/wiki/File:", "/wiki/Category:",
                                         "/wiki/Template:", "/wiki/Help:", "/wiki/Talk:"]):
                continue

            # Skip external wiki links
            if href.startswith("http") and self.base_url not in href:
                continue

            # Extract the page path
            if "/wiki/" in href:
                page_path = "/wiki/" + href.split("/wiki/")[-1]
                # Remove anchor
                if "#" in page_path:
                    page_path = page_path.split("#")[0]

                if page_path not in seen:
                    seen.add(page_path)
                    internal_links.append(page_path)

        return internal_links

    def _title_to_url(self, title: str) -> str:
        """Convert page title to URL"""
        encoded_title = quote(title.replace(" ", "_"))
        return f"{self.base_url}/wiki/{encoded_title}"

    def _url_to_title(self, url: str) -> str:
        """Extract page title from URL"""
        if "/wiki/" in url:
            title = url.split("/wiki/")[-1]
        else:
            title = url.split("/")[-1]
        return title.replace("_", " ")

    def _clean_content(self, text: str) -> str:
        """Clean extracted text content"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Remove common wiki artifacts
        text = re.sub(r'\[edit\]', '', text)
        text = re.sub(r'\[\d+\]', '', text)  # Reference numbers

        return text.strip()


# Convenience functions for common wikis
def create_mediawiki_scraper(url: str, **kwargs) -> MediaWikiScraper:
    """Create a scraper for any MediaWiki site"""
    return MediaWikiScraper(url, **kwargs)


# Quick test
if __name__ == "__main__":
    print("Testing MediaWiki scraper...")

    # Test with Appropedia
    print("\n=== Testing Appropedia ===")
    scraper = MediaWikiScraper("https://www.appropedia.org", api_path="/w/api.php")
    info = scraper.get_site_info()
    if info:
        print(f"Site: {info.get('general', {}).get('sitename')}")
        print(f"Articles: {info.get('statistics', {}).get('articles')}")

    # Test with Fandom
    print("\n=== Testing Solar Cooking Fandom ===")
    scraper = MediaWikiScraper("https://solarcooking.fandom.com")
    info = scraper.get_site_info()
    if info:
        print(f"Site: {info.get('general', {}).get('sitename')}")
        print(f"Articles: {info.get('statistics', {}).get('articles')}")

    # Test search
    print("\nSearching for 'parabolic'...")
    urls = scraper.search_pages("parabolic", limit=5)
    for url in urls[:3]:
        print(f"  - {url}")
