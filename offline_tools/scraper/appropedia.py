"""
Appropedia.org scraper - appropriate technology wiki
https://www.appropedia.org/
"""

from .base import BaseScraper, ScrapedPage
from typing import List, Optional
from datetime import datetime
import re


class ApropediaScraper(BaseScraper):
    """Scraper for Appropedia wiki"""

    def __init__(self, rate_limit: float = 1.0):
        super().__init__(
            source_name="appropedia",
            base_url="https://www.appropedia.org",
            rate_limit=rate_limit
        )
        # Categories relevant to disaster/DIY content
        self.priority_categories = [
            "Water",
            "Sanitation",
            "Shelter",
            "Food_and_agriculture",
            "Energy",
            "Health",
            "Emergency_management",
            "Appropriate_technology",
            "How_tos"
        ]

    def search_pages(self, search_term: str, limit: Optional[int] = None) -> List[str]:
        """
        Search for pages matching a term using MediaWiki API.
        Useful for finding all pages related to a topic (e.g., "Hexayurt").

        Args:
            search_term: Term to search for
            limit: Maximum number of pages to return

        Returns:
            List of page URLs
        """
        urls = []
        api_url = f"{self.base_url}/w/api.php"
        offset = 0
        batch_limit = 100  # API max per request

        while True:
            if limit and len(urls) >= limit:
                break

            params = {
                "action": "query",
                "list": "search",
                "srsearch": search_term,
                "srlimit": batch_limit,
                "sroffset": offset,
                "format": "json"
            }

            self._wait_for_rate_limit()
            try:
                response = self.session.get(api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                search_results = data.get("query", {}).get("search", [])
                if not search_results:
                    break

                for result in search_results:
                    page_title = result.get("title", "")
                    page_url = f"{self.base_url}/wiki/{page_title.replace(' ', '_')}"
                    if page_url not in urls:
                        urls.append(page_url)

                        if limit and len(urls) >= limit:
                            break

                # Check if there are more results
                if "continue" not in data:
                    break
                offset = data["continue"].get("sroffset", offset + batch_limit)

            except Exception as e:
                print(f"Error searching for '{search_term}': {e}")
                break

        print(f"Found {len(urls)} pages matching '{search_term}'")
        return urls[:limit] if limit else urls

    def get_page_list(self, limit: Optional[int] = None,
                      categories: Optional[List[str]] = None) -> List[str]:
        """
        Get list of article URLs from Appropedia.
        Uses the MediaWiki API for efficiency.
        """
        urls = []
        categories = categories or self.priority_categories

        for category in categories:
            if limit and len(urls) >= limit:
                break

            # Use MediaWiki API to list category members
            api_url = f"{self.base_url}/w/api.php"
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmlimit": "500",  # Max per request
                "cmtype": "page",
                "format": "json"
            }

            self._wait_for_rate_limit()
            try:
                response = self.session.get(api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                for member in data.get("query", {}).get("categorymembers", []):
                    page_title = member.get("title", "")
                    # Skip talk pages, user pages, etc.
                    if ":" not in page_title or page_title.startswith("Category:"):
                        page_url = f"{self.base_url}/wiki/{page_title.replace(' ', '_')}"
                        if page_url not in urls:
                            urls.append(page_url)

                            if limit and len(urls) >= limit:
                                break

            except Exception as e:
                print(f"Error fetching category {category}: {e}")

        print(f"Found {len(urls)} pages to scrape from Appropedia")
        return urls[:limit] if limit else urls

    def scrape_page(self, url: str) -> Optional[ScrapedPage]:
        """Scrape a single Appropedia article using MediaWiki API"""
        # Extract page title from URL
        title = url.split("/wiki/")[-1].replace("_", " ")
        return self.scrape_page_by_title(title)

    def scrape_page_by_title(self, title: str) -> Optional[ScrapedPage]:
        """
        Fetch page content using MediaWiki API (more reliable than HTML scraping).
        """
        api_url = f"{self.base_url}/w/api.php"

        # Get page content as plain text
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|categories|revisions",
            "explaintext": "true",  # Plain text, no HTML
            "exsectionformat": "plain",
            "cllimit": "10",  # Limit categories
            "rvprop": "timestamp",
            "format": "json"
        }

        self._wait_for_rate_limit()
        try:
            response = self.session.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None

            # Get first (only) page
            page_data = list(pages.values())[0]

            # Check if page exists
            if "missing" in page_data:
                return None

            page_id = page_data.get("pageid")
            page_title = page_data.get("title", title)
            content = page_data.get("extract", "")

            if len(content) < 100:  # Skip stub articles
                return None

            # Get categories
            categories = []
            for cat in page_data.get("categories", []):
                cat_name = cat.get("title", "").replace("Category:", "")
                if cat_name:
                    categories.append(cat_name)

            # Get last modified
            last_modified = None
            revisions = page_data.get("revisions", [])
            if revisions:
                last_modified = revisions[0].get("timestamp")

            # Build URL
            page_url = f"{self.base_url}/wiki/{page_title.replace(' ', '_')}"

            # Clean content
            content = self._clean_content(content)

            return ScrapedPage(
                url=page_url,
                title=page_title,
                content=content,
                source=self.source_name,
                categories=categories[:10],
                last_modified=last_modified,
                content_hash=self._hash_content(content),
                scraped_at=datetime.utcnow().isoformat()
            )

        except Exception as e:
            print(f"Error fetching '{title}' via API: {e}")
            return None

    def _clean_content(self, text: str) -> str:
        """Clean extracted text content"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Remove common wiki artifacts
        text = re.sub(r'\[edit\]', '', text)
        text = re.sub(r'\[\d+\]', '', text)  # Reference numbers

        return text.strip()


# Quick test
if __name__ == "__main__":
    scraper = ApropediaScraper(rate_limit=1.5)

    # Test getting page list
    urls = scraper.get_page_list(limit=5)
    print(f"Sample URLs: {urls}")

    # Test scraping one page
    if urls:
        page = scraper.scrape_page(urls[0])
        if page:
            print(f"\nScraped: {page.title}")
            print(f"Content length: {len(page.content)} chars")
            print(f"Categories: {page.categories}")
