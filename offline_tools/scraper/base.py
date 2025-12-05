"""
Base scraper class for wiki/content sites.
All site-specific scrapers inherit from this.
"""

import requests
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import time
import hashlib


@dataclass
class ScrapedPage:
    """Represents a scraped page with metadata"""
    url: str
    title: str
    content: str  # Clean text content
    source: str   # e.g., "appropedia", "akvopedia"
    categories: List[str]
    last_modified: Optional[str]
    content_hash: str  # For change detection
    scraped_at: str

    def to_dict(self) -> Dict:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "categories": self.categories,
            "last_modified": self.last_modified,
            "content_hash": self.content_hash,
            "scraped_at": self.scraped_at
        }


class RateLimitMixin:
    """
    Mixin class for rate-limited HTTP requests.
    Use this for any scraper that needs rate limiting without inheriting from BaseScraper.
    """

    def _init_rate_limiter(self, rate_limit: float = 1.0, user_agent: Optional[str] = None):
        """Initialize rate limiting and HTTP session"""
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })

    def _wait_for_rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _hash_content(self, content: str) -> str:
        """Generate hash for change detection"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()


class BaseScraper(RateLimitMixin, ABC):
    """Abstract base class for content scrapers"""

    def __init__(self, source_name: str, base_url: str, rate_limit: float = 1.0):
        """
        Args:
            source_name: Identifier for this source (e.g., "appropedia")
            base_url: Base URL of the site
            rate_limit: Minimum seconds between requests
        """
        self.source_name = source_name
        self.base_url = base_url
        self._init_rate_limiter(rate_limit)

    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a page with rate limiting"""
        self._wait_for_rate_limit()
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'lxml')
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    @abstractmethod
    def get_page_list(self, limit: Optional[int] = None) -> List[str]:
        """
        Get list of page URLs to scrape.
        Override in subclass with site-specific logic.
        """
        pass

    @abstractmethod
    def scrape_page(self, url: str) -> Optional[ScrapedPage]:
        """
        Scrape a single page and return structured data.
        Override in subclass with site-specific parsing.
        """
        pass

    def scrape_all(self, limit: Optional[int] = None,
                   progress_callback=None) -> List[ScrapedPage]:
        """
        Scrape all pages from this source.

        Args:
            limit: Maximum number of pages to scrape (None = all)
            progress_callback: Optional function(current, total) for progress updates

        Returns:
            List of ScrapedPage objects
        """
        urls = self.get_page_list(limit=limit)
        total = len(urls)
        pages = []

        for i, url in enumerate(urls):
            if progress_callback:
                progress_callback(i + 1, total)

            page = self.scrape_page(url)
            if page:
                pages.append(page)

        return pages
