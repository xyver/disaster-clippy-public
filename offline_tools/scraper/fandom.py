"""
Fandom Wiki scraper - extends MediaWikiScraper.
Works with any Fandom wiki (*.fandom.com).

Usage:
    # Solar Cooking Wiki
    scraper = FandomScraper("solarcooking")
    pages = scraper.search_pages("box cooker", limit=20)

    # Any other Fandom wiki
    scraper = FandomScraper("survivalguide")
    pages = scraper.get_all_pages(limit=100)
"""

from typing import Optional

from .mediawiki import MediaWikiScraper


class FandomScraper(MediaWikiScraper):
    """
    Scraper for Fandom wikis (*.fandom.com).
    Thin wrapper around MediaWikiScraper with Fandom-specific defaults.
    """

    def __init__(self, wiki_name: str, rate_limit: float = 1.0):
        """
        Args:
            wiki_name: The wiki subdomain (e.g., "solarcooking" for solarcooking.fandom.com)
            rate_limit: Minimum seconds between requests
        """
        base_url = f"https://{wiki_name}.fandom.com"
        super().__init__(
            base_url=base_url,
            source_name=wiki_name,
            api_path="/api.php",  # Fandom uses standard /api.php
            rate_limit=rate_limit
        )
        self.wiki_name = wiki_name


# Convenience function for quick access
def create_fandom_scraper(wiki_name: str, rate_limit: float = 1.0) -> FandomScraper:
    """
    Create a scraper for any Fandom wiki.

    Examples:
        scraper = create_fandom_scraper("solarcooking")  # Solar Cooking Wiki
        scraper = create_fandom_scraper("survival")      # Survival Wiki
        scraper = create_fandom_scraper("diy")           # DIY Wiki
    """
    return FandomScraper(wiki_name, rate_limit)


# Quick test
if __name__ == "__main__":
    print("Testing Fandom scraper with Solar Cooking Wiki...")
    scraper = FandomScraper("solarcooking", rate_limit=1.0)

    # Get site info
    info = scraper.get_site_info()
    if info:
        print(f"Site: {info.get('general', {}).get('sitename')}")
        print(f"Articles: {info.get('statistics', {}).get('articles')}")

    # Test search
    print("\nSearching for 'box cooker'...")
    urls = scraper.search_pages("box cooker", limit=5)
    for url in urls:
        print(f"  - {url}")

    # Test page scraping
    if urls:
        print(f"\nScraping first result...")
        page = scraper.scrape_page(urls[0])
        if page:
            print(f"  Title: {page.title}")
            print(f"  Content length: {len(page.content)} chars")
            print(f"  Categories: {page.categories}")
        else:
            print("  Failed to scrape page")

    # Test category listing
    print("\nListing categories...")
    cats = scraper.list_categories(limit=10)
    for cat in cats:
        print(f"  - {cat}")
