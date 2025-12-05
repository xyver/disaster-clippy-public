from .base import BaseScraper, ScrapedPage, RateLimitMixin
from .appropedia import ApropediaScraper
from .mediawiki import MediaWikiScraper, create_mediawiki_scraper
from .fandom import FandomScraper, create_fandom_scraper
from .static_site import StaticSiteScraper, create_static_scraper, BuildItSolarScraper
from .pdf import PDFScraper, create_pdf_scraper
from .substack import SubstackScraper, create_substack_scraper

# Scraper registry for CLI/programmatic access
SCRAPER_REGISTRY = {
    # MediaWiki-based scrapers
    "mediawiki": MediaWikiScraper,
    "appropedia": ApropediaScraper,
    "fandom": FandomScraper,
    # Static site scrapers
    "static": StaticSiteScraper,
    "builditsolar": BuildItSolarScraper,
    # Document scrapers
    "pdf": PDFScraper,
    # Newsletter scrapers
    "substack": SubstackScraper,
}


def get_scraper(scraper_type: str, **kwargs):
    """
    Factory function to get the appropriate scraper.

    Args:
        scraper_type: One of 'mediawiki', 'appropedia', 'fandom', 'static',
                      'builditsolar', 'pdf'
        **kwargs: Arguments passed to scraper constructor

    Returns:
        Scraper instance

    Examples:
        # Generic MediaWiki site
        scraper = get_scraper("mediawiki", base_url="https://wiki.example.com")

        # Specific preset
        scraper = get_scraper("appropedia")

        # Fandom wiki
        scraper = get_scraper("fandom", wiki_name="solarcooking")

        # Static site
        scraper = get_scraper("static", base_url="https://www.builditsolar.com")

        # PDF scraper
        scraper = get_scraper("pdf", source_name="research")
    """
    scraper_type = scraper_type.lower()

    if scraper_type not in SCRAPER_REGISTRY:
        available = ", ".join(SCRAPER_REGISTRY.keys())
        raise ValueError(f"Unknown scraper type: {scraper_type}. Available: {available}")

    scraper_class = SCRAPER_REGISTRY[scraper_type]
    return scraper_class(**kwargs)


def list_scrapers():
    """List all available scraper types"""
    return list(SCRAPER_REGISTRY.keys())


__all__ = [
    # Base classes
    'BaseScraper',
    'ScrapedPage',
    'RateLimitMixin',
    # Specific scrapers
    'ApropediaScraper',
    'MediaWikiScraper',
    'create_mediawiki_scraper',
    'FandomScraper',
    'create_fandom_scraper',
    'StaticSiteScraper',
    'create_static_scraper',
    'BuildItSolarScraper',
    'PDFScraper',
    'create_pdf_scraper',
    'SubstackScraper',
    'create_substack_scraper',
    # Factory functions
    'get_scraper',
    'list_scrapers',
    'SCRAPER_REGISTRY',
]
