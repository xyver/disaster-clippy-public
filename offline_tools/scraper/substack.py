"""
Substack Newsletter Scraper.
Ingests posts from a Substack newsletter using a CSV export and optional session cookie for paid content.

Usage:
    # Basic usage with CSV file
    scraper = SubstackScraper(
        csv_path="posts.csv",
        newsletter_url="https://thebarracks.substack.com"
    )

    # With authentication for paid posts
    scraper = SubstackScraper(
        csv_path="posts.csv",
        newsletter_url="https://thebarracks.substack.com",
        session_cookie="your_substack.sid_cookie"
    )

    # Get all published posts
    urls = scraper.get_page_list()

    # Scrape a single post
    page = scraper.scrape_page(url)
"""

import csv
import os
import re
from typing import List, Optional
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from .base import BaseScraper, ScrapedPage


class SubstackScraper(BaseScraper):
    """
    Scraper for Substack newsletters.
    Uses a CSV export (from Substack settings) to get post list,
    then fetches each post's content via HTTP.
    """

    def __init__(
        self,
        csv_path: str,
        newsletter_url: str,
        source_name: Optional[str] = None,
        session_cookie: Optional[str] = None,
        include_paid: bool = True,
        rate_limit: float = 1.0
    ):
        """
        Args:
            csv_path: Path to posts.csv exported from Substack
            newsletter_url: Base URL of the newsletter (e.g., "https://thebarracks.substack.com")
            source_name: Identifier for this source (auto-detected from URL if not provided)
            session_cookie: Value of 'substack.sid' cookie for accessing paid content
            include_paid: Whether to include paid posts (requires session_cookie)
            rate_limit: Minimum seconds between requests
        """
        # Auto-detect source name from URL
        if source_name is None:
            parsed = urlparse(newsletter_url)
            # Extract subdomain (e.g., "thebarracks" from "thebarracks.substack.com")
            source_name = parsed.netloc.split(".")[0]

        super().__init__(
            source_name=source_name,
            base_url=newsletter_url.rstrip("/"),
            rate_limit=rate_limit
        )

        self.csv_path = csv_path
        self.session_cookie = session_cookie or os.getenv("SUBSTACK_SESSION_COOKIE")
        self.include_paid = include_paid

        # Add session cookie to requests if available
        if self.session_cookie:
            self.session.cookies.set("substack.sid", self.session_cookie)

        # Store parsed posts from CSV
        self._posts = None

    def _load_posts_csv(self) -> List[dict]:
        """Load and parse the posts CSV file"""
        if self._posts is not None:
            return self._posts

        self._posts = []

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only include published posts
                if row.get('is_published') != 'true':
                    continue

                # Skip paid posts if no auth or not requested
                audience = row.get('audience', 'everyone')
                if audience == 'only_paid' and (not self.session_cookie or not self.include_paid):
                    continue

                # Extract slug from post_id (format: "123456.slug-here")
                post_id = row.get('post_id', '')
                if '.' in post_id:
                    slug = post_id.split('.', 1)[1]
                else:
                    slug = post_id

                # Skip empty slugs
                if not slug:
                    continue

                self._posts.append({
                    'slug': slug,
                    'title': row.get('title', ''),
                    'subtitle': row.get('subtitle', ''),
                    'post_date': row.get('post_date', ''),
                    'audience': audience,
                    'type': row.get('type', 'newsletter'),
                    'url': f"{self.base_url}/p/{slug}"
                })

        return self._posts

    def get_page_list(self, limit: Optional[int] = None) -> List[str]:
        """Get list of post URLs from CSV"""
        posts = self._load_posts_csv()
        urls = [p['url'] for p in posts]

        if limit:
            return urls[:limit]
        return urls

    def get_post_metadata(self, url: str) -> Optional[dict]:
        """Get metadata for a post from the CSV data"""
        posts = self._load_posts_csv()
        for post in posts:
            if post['url'] == url:
                return post
        return None

    def scrape_page(self, url: str) -> Optional[ScrapedPage]:
        """Scrape a single Substack post"""
        soup = self._fetch_page(url)
        if not soup:
            return None

        # Check for paywall
        paywall_indicators = [
            'Keep reading with a 7-day free trial',
            'This post is for paid subscribers',
            'Subscribe to continue reading'
        ]
        page_text = soup.get_text()
        for indicator in paywall_indicators:
            if indicator in page_text:
                print(f"Paywall detected for {url} - skipping")
                return None

        # Get metadata from CSV if available
        metadata = self.get_post_metadata(url)

        # Extract title
        title = "Unknown"
        if metadata and metadata.get('title'):
            title = metadata['title']
        else:
            # Try page elements
            title_elem = soup.find('h1', class_='post-title') or soup.find('h1')
            if title_elem:
                title = title_elem.get_text(strip=True)

        # Find main content
        content_elem = None

        # Substack content selectors (in order of preference)
        selectors = [
            ('div', {'class': 'body'}),
            ('div', {'class': 'available-content'}),
            ('div', {'class': 'post-content'}),
            ('article', {}),
        ]

        for tag, attrs in selectors:
            content_elem = soup.find(tag, attrs) if attrs else soup.find(tag)
            if content_elem:
                break

        if not content_elem:
            print(f"Could not find content element for {url}")
            return None

        # Remove unwanted elements
        for selector in ['script', 'style', 'nav', 'footer', '.subscription-widget',
                         '.post-footer', '.comments-section', '.share-buttons']:
            if selector.startswith('.'):
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

        # Skip very short content (likely paywall or error)
        if len(content) < 200:
            print(f"Content too short ({len(content)} chars) for {url} - skipping")
            return None

        # Build categories
        categories = ['newsletter', self.source_name]
        if metadata:
            if metadata.get('type'):
                categories.append(metadata['type'])
            if metadata.get('audience') == 'only_paid':
                categories.append('paid')

        # Get post date
        last_modified = None
        if metadata and metadata.get('post_date'):
            last_modified = metadata['post_date']
        else:
            # Try to find date in page
            date_elem = soup.find('time') or soup.find(class_='post-date')
            if date_elem:
                last_modified = date_elem.get('datetime') or date_elem.get_text(strip=True)

        return ScrapedPage(
            url=url,
            title=title,
            content=content,
            source=self.source_name,
            categories=categories,
            last_modified=last_modified,
            content_hash=self._hash_content(content),
            scraped_at=datetime.utcnow().isoformat()
        )

    def scrape_all(self, limit: Optional[int] = None,
                   progress_callback=None) -> List[ScrapedPage]:
        """
        Scrape all posts from the newsletter.

        Args:
            limit: Maximum number of posts to scrape (None = all)
            progress_callback: Optional function(current, total) for progress updates

        Returns:
            List of ScrapedPage objects
        """
        urls = self.get_page_list(limit=limit)
        total = len(urls)
        pages = []
        skipped = 0

        print(f"Scraping {total} posts from {self.source_name}...")
        if self.session_cookie:
            print("(Using session cookie for paid content)")

        for i, url in enumerate(urls):
            if progress_callback:
                progress_callback(i + 1, total)

            page = self.scrape_page(url)
            if page:
                pages.append(page)
            else:
                skipped += 1

        print(f"Scraped {len(pages)} posts, skipped {skipped}")
        return pages


# Convenience function
def create_substack_scraper(csv_path: str, newsletter_url: str, **kwargs) -> SubstackScraper:
    """Create a Substack scraper"""
    return SubstackScraper(csv_path=csv_path, newsletter_url=newsletter_url, **kwargs)


# Quick test
if __name__ == "__main__":
    import sys

    print("Testing Substack Scraper...")

    # Check for CSV file
    csv_path = "posts.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    # Load environment for cookie
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    scraper = SubstackScraper(
        csv_path=csv_path,
        newsletter_url="https://thebarracks.substack.com"
    )

    # Test getting post list
    print("\nGetting posts from CSV...")
    urls = scraper.get_page_list(limit=5)
    print(f"Found {len(urls)} posts (showing first 5)")
    for url in urls[:5]:
        print(f"  - {url}")

    # Test scraping a single post
    if urls:
        print(f"\nScraping: {urls[0]}")
        page = scraper.scrape_page(urls[0])
        if page:
            print(f"  Title: {page.title}")
            print(f"  Content length: {len(page.content)} chars")
            print(f"  Categories: {page.categories}")
            print(f"  Date: {page.last_modified}")
            print(f"  First 300 chars: {page.content[:300]}...")
        else:
            print("  Failed to scrape page")
