"""
PDF Scraper/Ingester for academic papers and documents.

Supports:
- Local PDF files
- Folders of PDFs
- PDF URLs (downloads and processes)
- Website crawling to find PDFs

Usage:
    scraper = PDFScraper(source_name="research_papers")

    # Single file
    doc = scraper.process_file("paper.pdf")

    # Folder
    docs = scraper.process_folder("/path/to/pdfs")

    # URL
    doc = scraper.process_url("https://example.com/paper.pdf")

    # Crawl website for PDFs
    docs = scraper.crawl_for_pdfs("https://example.com/papers/", limit=50)
"""

import re
import hashlib
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime
from urllib.parse import urlparse, urljoin
import requests

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from bs4 import BeautifulSoup

from .base import ScrapedPage


def parse_pdf_date(pdf_date: str) -> Optional[str]:
    """
    Convert PDF date format to ISO 8601 format.

    PDF dates look like: D:20190215090811-05'00' or D:20180101120000Z
    Returns ISO format: 2019-02-15T09:08:11 or None if parsing fails
    """
    if not pdf_date:
        return None

    # Remove 'D:' prefix if present
    date_str = pdf_date
    if date_str.startswith("D:"):
        date_str = date_str[2:]

    # Remove timezone suffix (we'll just use the base datetime)
    # Format variations: -05'00', +00'00', Z
    date_str = re.sub(r"[+-]\d{2}'\d{2}'$", "", date_str)
    date_str = re.sub(r"Z$", "", date_str)

    try:
        # Parse: YYYYMMDDHHmmss
        if len(date_str) >= 14:
            dt = datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
            return dt.isoformat()
        # Parse: YYYYMMDD
        elif len(date_str) >= 8:
            dt = datetime.strptime(date_str[:8], "%Y%m%d")
            return dt.isoformat()
        # Parse: YYYY
        elif len(date_str) >= 4:
            return f"{date_str[:4]}-01-01T00:00:00"
    except ValueError:
        pass

    return None


class PDFScraper:
    """
    Scraper for PDF documents - academic papers, reports, manuals, etc.
    Uses RateLimitMixin for consistent rate limiting across all scrapers.
    """

    def __init__(self, source_name: str = "pdf", rate_limit: float = 1.0):
        """
        Args:
            source_name: Identifier for this source (e.g., "research_papers")
            rate_limit: Minimum seconds between web requests
        """
        from .base import RateLimitMixin

        self.source_name = source_name

        # Use mixin for rate limiting (manually since we don't inherit from BaseScraper)
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; DisasterClippy/1.0; +https://github.com/disaster-clippy)"
        })

        # Check available PDF libraries
        if not HAS_PYMUPDF and not HAS_PYPDF:
            raise ImportError(
                "No PDF library available. Install one of:\n"
                "  pip install pymupdf  (recommended)\n"
                "  pip install pypdf"
            )

    def _wait_for_rate_limit(self):
        """Respect rate limiting for web requests"""
        import time
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _hash_content(self, content: str) -> str:
        """Generate hash of content for deduplication"""
        return hashlib.md5(content.encode()).hexdigest()

    def _extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (fitz)"""
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)

    def _extract_text_pypdf(self, pdf_path: str) -> str:
        """Extract text using pypdf"""
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n".join(text_parts)

    def _extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF using available library"""
        if HAS_PYMUPDF:
            return self._extract_text_pymupdf(pdf_path)
        elif HAS_PYPDF:
            return self._extract_text_pypdf(pdf_path)
        else:
            raise RuntimeError("No PDF library available")

    def _extract_metadata_pymupdf(self, pdf_path: str) -> Dict:
        """Extract metadata using PyMuPDF"""
        doc = fitz.open(pdf_path)
        metadata = doc.metadata or {}
        doc.close()
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "keywords": metadata.get("keywords", ""),
            "creator": metadata.get("creator", ""),
            "creation_date": metadata.get("creationDate", ""),
        }

    def _extract_metadata_pypdf(self, pdf_path: str) -> Dict:
        """Extract metadata using pypdf"""
        reader = PdfReader(pdf_path)
        metadata = reader.metadata or {}
        return {
            "title": metadata.get("/Title", ""),
            "author": metadata.get("/Author", ""),
            "subject": metadata.get("/Subject", ""),
            "keywords": metadata.get("/Keywords", ""),
            "creator": metadata.get("/Creator", ""),
            "creation_date": str(metadata.get("/CreationDate", "")),
        }

    def _extract_metadata(self, pdf_path: str) -> Dict:
        """Extract PDF metadata using available library"""
        try:
            if HAS_PYMUPDF:
                return self._extract_metadata_pymupdf(pdf_path)
            elif HAS_PYPDF:
                return self._extract_metadata_pypdf(pdf_path)
        except Exception as e:
            print(f"Error extracting metadata: {e}")
        return {}

    def _parse_academic_content(self, text: str, metadata: Dict) -> Dict:
        """
        Parse academic paper structure to extract title, abstract, etc.

        Returns dict with: title, abstract, authors, content (main body)
        """
        lines = text.split("\n")
        result = {
            "title": metadata.get("title", ""),
            "authors": metadata.get("author", ""),
            "abstract": "",
            "content": text,
            "keywords": []
        }

        # Try to extract title from first non-empty lines if not in metadata
        if not result["title"]:
            for line in lines[:10]:
                line = line.strip()
                # Title is usually the first substantial line (not too short, not too long)
                if len(line) > 10 and len(line) < 200:
                    # Skip lines that look like headers/footers
                    if not any(skip in line.lower() for skip in
                              ["page", "vol.", "volume", "journal", "copyright", "doi:"]):
                        result["title"] = line
                        break

        # Try to extract abstract
        abstract_patterns = [
            r"abstract[:\s]*\n?(.*?)(?=\n\s*(?:introduction|keywords|1\.|1\s|background))",
            r"summary[:\s]*\n?(.*?)(?=\n\s*(?:introduction|keywords|1\.|1\s))",
        ]

        text_lower = text.lower()
        for pattern in abstract_patterns:
            match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
            if match:
                # Get the actual text (not lowercased) at the same position
                start, end = match.start(1), match.end(1)
                abstract = text[start:end].strip()
                # Clean up the abstract
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 100:  # Reasonable abstract length
                    result["abstract"] = abstract[:2000]  # Cap at 2000 chars
                    break

        # Try to extract keywords
        keywords_match = re.search(
            r"keywords?[:\s]*([^\n]+)",
            text_lower,
            re.IGNORECASE
        )
        if keywords_match:
            start, end = keywords_match.start(1), keywords_match.end(1)
            keywords_text = text[start:end]
            # Split by comma or semicolon
            keywords = re.split(r'[,;]', keywords_text)
            result["keywords"] = [kw.strip() for kw in keywords if kw.strip()][:10]

        # Clean up content - remove excessive whitespace
        result["content"] = re.sub(r'\n{3,}', '\n\n', text)
        result["content"] = re.sub(r' {2,}', ' ', result["content"])

        return result

    def _clean_filename_to_title(self, filename: str) -> str:
        """Convert filename to readable title"""
        # Remove extension
        name = Path(filename).stem
        # Replace underscores and hyphens with spaces
        name = re.sub(r'[-_]+', ' ', name)
        # Remove common prefixes like dates or IDs
        name = re.sub(r'^\d{4}[-_]?\d{0,2}[-_]?\d{0,2}[-_]?', '', name)
        # Title case
        return name.strip().title()

    def process_file(self, file_path: str, url: Optional[str] = None) -> Optional[ScrapedPage]:
        """
        Process a single PDF file.

        Args:
            file_path: Path to the PDF file
            url: Optional URL if this was downloaded from web

        Returns:
            ScrapedPage or None if processing fails
        """
        path = Path(file_path)
        if not path.exists():
            print(f"File not found: {file_path}")
            return None

        if not path.suffix.lower() == '.pdf':
            print(f"Not a PDF file: {file_path}")
            return None

        try:
            # Extract text and metadata
            text = self._extract_text(str(path))
            metadata = self._extract_metadata(str(path))

            if len(text) < 100:
                print(f"PDF has too little text: {file_path}")
                return None

            # Parse academic structure
            parsed = self._parse_academic_content(text, metadata)

            # Determine title
            title = parsed["title"] or self._clean_filename_to_title(path.name)

            # Build content with abstract if available
            content = parsed["content"]
            if parsed["abstract"]:
                content = f"Abstract: {parsed['abstract']}\n\n{content}"

            # Build URL (file path if no URL provided)
            doc_url = url or f"file://{path.absolute()}"

            # Categories from keywords
            categories = parsed["keywords"] or []
            if metadata.get("subject"):
                categories.append(metadata["subject"])

            return ScrapedPage(
                url=doc_url,
                title=title,
                content=content,
                source=self.source_name,
                categories=categories[:10],
                last_modified=parse_pdf_date(metadata.get("creation_date")),
                content_hash=self._hash_content(content),
                scraped_at=datetime.utcnow().isoformat()
            )

        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            return None

    def process_file_chunked(self, file_path: str, url: Optional[str] = None,
                              chunk_size: int = 4000, overlap: int = 200) -> List[ScrapedPage]:
        """
        Process a PDF file and split into chunks for better search results.

        Large PDFs (like 200K+ chars) should be chunked so that:
        1. Vector search can find specific relevant sections
        2. Each chunk gets its own embedding
        3. Users get more precise results

        Args:
            file_path: Path to the PDF file
            url: Optional URL if downloaded from web
            chunk_size: Target characters per chunk (default 4000)
            overlap: Characters to overlap between chunks (default 200)

        Returns:
            List of ScrapedPage objects, one per chunk
        """
        path = Path(file_path)
        if not path.exists():
            print(f"File not found: {file_path}")
            return []

        if not path.suffix.lower() == '.pdf':
            print(f"Not a PDF file: {file_path}")
            return []

        try:
            # Extract text and metadata
            text = self._extract_text(str(path))
            metadata = self._extract_metadata(str(path))

            if len(text) < 100:
                print(f"PDF has too little text: {file_path}")
                return []

            # Parse academic structure
            parsed = self._parse_academic_content(text, metadata)

            # Determine base title
            base_title = parsed["title"] or self._clean_filename_to_title(path.name)

            # Build base URL
            doc_url = url or f"file://{path.absolute()}"

            # Categories from keywords
            categories = parsed["keywords"] or []
            if metadata.get("subject"):
                categories.append(metadata["subject"])

            # If document is small enough, return as single page
            content = parsed["content"]
            if len(content) <= chunk_size * 1.5:
                single_page = ScrapedPage(
                    url=doc_url,
                    title=base_title,
                    content=content,
                    source=self.source_name,
                    categories=categories[:10],
                    last_modified=parse_pdf_date(metadata.get("creation_date")),
                    content_hash=self._hash_content(content),
                    scraped_at=datetime.utcnow().isoformat()
                )
                return [single_page]

            # Split into chunks - try to split on section headers or paragraphs
            chunks = self._smart_chunk_text(content, chunk_size, overlap)

            print(f"Split PDF into {len(chunks)} chunks (avg {sum(len(c) for c in chunks)//len(chunks)} chars each)")

            pages = []
            for i, chunk_text in enumerate(chunks, 1):
                # Try to extract a section title from the chunk
                section_title = self._extract_section_title(chunk_text)
                if section_title:
                    title = f"{base_title} - {section_title}"
                else:
                    title = f"{base_title} (Part {i}/{len(chunks)})"

                # Create unique URL for this chunk (add fragment)
                chunk_url = f"{doc_url}#chunk-{i}"

                page = ScrapedPage(
                    url=chunk_url,
                    title=title,
                    content=chunk_text,
                    source=self.source_name,
                    categories=categories[:10],
                    last_modified=parse_pdf_date(metadata.get("creation_date")),
                    content_hash=self._hash_content(chunk_text),
                    scraped_at=datetime.utcnow().isoformat()
                )
                pages.append(page)

            return pages

        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            return []

    def _smart_chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into chunks, trying to break at natural boundaries.

        Prioritizes breaking at:
        1. Section headers (roman numerals, numbered sections)
        2. Paragraph breaks (double newlines)
        3. Sentence boundaries
        4. Word boundaries (last resort)
        """
        chunks = []

        # Pattern to detect section headers
        section_pattern = re.compile(
            r'\n\s*(?:'
            r'(?:I{1,3}|IV|VI{0,3}|IX|X{1,3})\.'  # Roman numerals
            r'|[0-9]+\.[0-9]*'                      # Numbered sections
            r'|(?:Chapter|Section|Part)\s+[0-9]+'   # Named sections
            r'|[A-Z][A-Z\s]{5,50}(?:\n|$)'          # ALL CAPS HEADERS
            r')',
            re.IGNORECASE
        )

        # First try to split on major section headers
        sections = section_pattern.split(text)
        headers = section_pattern.findall(text)

        # Recombine headers with their content
        combined = []
        for i, section in enumerate(sections):
            if i > 0 and i - 1 < len(headers):
                combined.append(headers[i-1].strip() + "\n" + section)
            else:
                combined.append(section)

        # Now process each section
        for section in combined:
            if not section.strip():
                continue

            # If section fits in chunk, add it
            if len(section) <= chunk_size:
                if chunks and len(chunks[-1]) + len(section) <= chunk_size:
                    # Merge with previous chunk if fits
                    chunks[-1] += "\n\n" + section
                else:
                    chunks.append(section)
            else:
                # Section too big, split on paragraphs
                paragraphs = re.split(r'\n\s*\n', section)
                current_chunk = ""

                for para in paragraphs:
                    if not para.strip():
                        continue

                    if len(current_chunk) + len(para) <= chunk_size:
                        current_chunk += "\n\n" + para if current_chunk else para
                    else:
                        # Save current chunk with overlap
                        if current_chunk:
                            chunks.append(current_chunk)
                            # Start new chunk with overlap from end of previous
                            if overlap > 0:
                                current_chunk = current_chunk[-overlap:] + "\n\n" + para
                            else:
                                current_chunk = para
                        else:
                            # Paragraph itself is too big, split by sentences
                            sentences = re.split(r'(?<=[.!?])\s+', para)
                            for sent in sentences:
                                if len(current_chunk) + len(sent) <= chunk_size:
                                    current_chunk += " " + sent if current_chunk else sent
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk)
                                    current_chunk = sent

                if current_chunk:
                    chunks.append(current_chunk)

        # Clean up chunks
        chunks = [c.strip() for c in chunks if c.strip() and len(c.strip()) > 100]

        return chunks

    def _extract_section_title(self, chunk_text: str) -> Optional[str]:
        """
        Try to extract a section title from the beginning of a chunk.
        """
        lines = chunk_text.strip().split('\n')[:5]

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section header patterns
            # Roman numeral sections: "III. PANDEMIC RESPONSE"
            match = re.match(r'^((?:I{1,3}|IV|VI{0,3}|IX|X{1,3})\.\s*[\w\s]{3,50})$', line, re.IGNORECASE)
            if match:
                return match.group(1).strip()

            # Numbered sections: "3.2 Community Planning"
            match = re.match(r'^([0-9]+\.(?:[0-9]+\.?)?\s*[\w\s]{3,50})$', line)
            if match:
                return match.group(1).strip()

            # ALL CAPS headers (but not too long)
            if line.isupper() and 5 <= len(line) <= 50:
                return line.title()

            # Title case line that looks like a header
            if len(line) < 60 and not line.endswith('.') and line[0].isupper():
                words = line.split()
                if len(words) >= 2 and len(words) <= 8:
                    return line

        return None

    def process_folder(self, folder_path: str, recursive: bool = True) -> List[ScrapedPage]:
        """
        Process all PDFs in a folder.

        Args:
            folder_path: Path to folder containing PDFs
            recursive: Whether to search subdirectories

        Returns:
            List of ScrapedPage objects
        """
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Folder not found: {folder_path}")
            return []

        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(folder.glob(pattern))

        print(f"Found {len(pdf_files)} PDF files in {folder_path}")

        results = []
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"Processing {i}/{len(pdf_files)}: {pdf_path.name}")
            page = self.process_file(str(pdf_path))
            if page:
                results.append(page)

        print(f"Successfully processed {len(results)}/{len(pdf_files)} PDFs")
        return results

    def process_url(self, url: str) -> Optional[ScrapedPage]:
        """
        Download and process a PDF from URL.

        Args:
            url: URL to the PDF file

        Returns:
            ScrapedPage or None if processing fails
        """
        self._wait_for_rate_limit()

        try:
            response = self.session.get(url, timeout=60, stream=True)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
                print(f"URL does not appear to be a PDF: {url}")
                return None

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name

            # Process the downloaded file
            result = self.process_file(tmp_path, url=url)

            # Clean up temp file
            Path(tmp_path).unlink()

            return result

        except Exception as e:
            print(f"Error downloading PDF from {url}: {e}")
            return None

    def process_urls(self, urls: List[str]) -> List[ScrapedPage]:
        """
        Process multiple PDF URLs.

        Args:
            urls: List of PDF URLs

        Returns:
            List of ScrapedPage objects
        """
        results = []
        for i, url in enumerate(urls, 1):
            print(f"Processing {i}/{len(urls)}: {url}")
            page = self.process_url(url)
            if page:
                results.append(page)

        print(f"Successfully processed {len(results)}/{len(urls)} PDFs")
        return results

    def find_pdfs_on_page(self, page_url: str) -> List[str]:
        """
        Find all PDF links on a webpage.

        Args:
            page_url: URL of the webpage to scan

        Returns:
            List of PDF URLs found
        """
        self._wait_for_rate_limit()

        try:
            response = self.session.get(page_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            pdf_urls = set()

            # Find all links
            for link in soup.find_all("a", href=True):
                href = link["href"]

                # Check if it's a PDF
                if href.lower().endswith(".pdf"):
                    # Make absolute URL
                    full_url = urljoin(page_url, href)
                    pdf_urls.add(full_url)

            return list(pdf_urls)

        except Exception as e:
            print(f"Error scanning page {page_url}: {e}")
            return []

    def crawl_for_pdfs(self, start_url: str, limit: int = 50,
                       follow_links: bool = True, max_depth: int = 2,
                       url_pattern: Optional[str] = None) -> List[ScrapedPage]:
        """
        Crawl a website to find and process PDFs.

        Args:
            start_url: Starting URL for crawl
            limit: Maximum number of PDFs to process
            follow_links: Whether to follow links to find more pages
            max_depth: Maximum crawl depth
            url_pattern: Optional regex pattern to filter URLs

        Returns:
            List of ScrapedPage objects
        """
        parsed_start = urlparse(start_url)
        base_domain = parsed_start.netloc

        pdf_urls: Set[str] = set()
        visited_pages: Set[str] = set()
        pages_to_visit = [(start_url, 0)]  # (url, depth)

        print(f"Crawling {start_url} for PDFs (limit: {limit})...")

        while pages_to_visit and len(pdf_urls) < limit:
            current_url, depth = pages_to_visit.pop(0)

            if current_url in visited_pages:
                continue
            visited_pages.add(current_url)

            # Find PDFs on this page
            found_pdfs = self.find_pdfs_on_page(current_url)

            for pdf_url in found_pdfs:
                # Apply URL pattern filter if specified
                if url_pattern and not re.search(url_pattern, pdf_url):
                    continue
                pdf_urls.add(pdf_url)
                if len(pdf_urls) >= limit:
                    break

            print(f"  Found {len(found_pdfs)} PDFs on {current_url} (total: {len(pdf_urls)})")

            # Follow links if enabled and not at max depth
            if follow_links and depth < max_depth:
                self._wait_for_rate_limit()
                try:
                    response = self.session.get(current_url, timeout=30)
                    soup = BeautifulSoup(response.text, "html.parser")

                    for link in soup.find_all("a", href=True):
                        href = link["href"]
                        full_url = urljoin(current_url, href)
                        parsed = urlparse(full_url)

                        # Only follow links on same domain
                        if parsed.netloc == base_domain:
                            # Skip PDFs (we already found them)
                            if not full_url.lower().endswith(".pdf"):
                                if full_url not in visited_pages:
                                    pages_to_visit.append((full_url, depth + 1))
                except Exception as e:
                    print(f"  Error following links on {current_url}: {e}")

        print(f"Found {len(pdf_urls)} total PDF URLs, processing...")

        # Process the PDFs we found
        return self.process_urls(list(pdf_urls)[:limit])


    # ========================================================================
    # PAGE-AWARE AND SECTION-BASED EXTRACTION (for building codes, manuals)
    # ========================================================================

    def _extract_pages_pymupdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract text page-by-page using PyMuPDF, preserving page numbers.

        Returns:
            List of dicts: [{'page': 1, 'text': '...', 'char_count': N}, ...]
        """
        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            pages.append({
                'page': i + 1,  # 1-indexed page numbers
                'text': text,
                'char_count': len(text)
            })
        doc.close()
        return pages

    def _extract_pages_pypdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract text page-by-page using pypdf, preserving page numbers.
        """
        reader = PdfReader(pdf_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({
                'page': i + 1,
                'text': text,
                'char_count': len(text)
            })
        return pages

    def _extract_pages(self, pdf_path: str) -> List[Dict]:
        """Extract text page-by-page using available library."""
        if HAS_PYMUPDF:
            return self._extract_pages_pymupdf(pdf_path)
        elif HAS_PYPDF:
            return self._extract_pages_pypdf(pdf_path)
        else:
            raise RuntimeError("No PDF library available")

    def _detect_section_headers(self, text: str, page_num: int) -> List[Dict]:
        """
        Detect section headers in text with their positions.

        Handles patterns like:
        - "1 Section Title" or "1.1 Subsection" (on same line)
        - "1\nSection Title" (number on own line, title on next)
        - "Section 1: Title"
        - "CHAPTER 1"
        - "A.1 Appendix section"

        Returns:
            List of dicts: [{'number': '1.1', 'title': 'Section Title',
                           'level': 2, 'pos': 123, 'page': 1}, ...]
        """
        headers = []

        # Skip patterns - common page headers/footers to ignore
        skip_patterns = [
            r'^\d{4}\s+FORTIFIED',  # "2025 FORTIFIED Home..."
            r'^Page\s*\|',  # "Page | 5"
            r'^\d+\s*$',  # Lone numbers (often page numbers)
        ]

        def should_skip(line: str) -> bool:
            """Check if line matches skip patterns."""
            for pattern in skip_patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    return True
            return False

        def is_toc_title(title: str) -> bool:
            """Check if title looks like a TOC entry (has dots followed by page number)."""
            # Pattern: "Title ............. 5" or "Title ..... 10"
            if re.search(r'\.{3,}\s*\d+\s*$', title):
                return True
            # Also check for very long runs of dots
            if re.search(r'\.{10,}', title):
                return True
            return False

        # Pattern 1: Numbered sections on same line: "1.2.3 Title"
        # Must have proper section number followed by meaningful title
        numbered_same_line = re.compile(
            r'^[ \t]*'
            r'(\d+(?:\.\d+)+)'  # Section number with at least one dot: 1.1, 1.2.3
            r'[ \t]+'
            r'([A-Z][A-Za-z][^\n]{2,80})'  # Title starting with caps, at least 4 chars
            r'[ \t]*$',
            re.MULTILINE
        )

        # Pattern 2: Top-level section number on own line, title on next line
        # Common in FORTIFIED-style documents: "1\nProgram Overview"
        numbered_split_line = re.compile(
            r'^[ \t]*'
            r'(\d+)'  # Just the number: 1, 2, 3
            r'[ \t]*\n'  # End of line, newline
            r'[ \t]*'
            r'([A-Z][A-Za-z][^\n]{2,200})',  # Title on next line (capture more for TOC check)
            re.MULTILINE
        )

        # Pattern 3: Subsection with split lines: "1.1\nSubsection Title"
        subsection_split_line = re.compile(
            r'^[ \t]*'
            r'(\d+\.\d+(?:\.\d+)*)'  # Number with dots
            r'[ \t]*\n'
            r'[ \t]*'
            r'([A-Z][A-Za-z][^\n]{2,200})',  # Capture more to check for TOC dots
            re.MULTILINE
        )

        # Pattern 4: "Section X" or "Chapter X" style
        named_pattern = re.compile(
            r'^[ \t]*'
            r'((?:Section|Chapter|Part|Appendix)\s+[\dA-Z]+(?:\.\d+)*)'
            r'[:\.\s]+'
            r'([A-Z][^\n]{3,80})?'
            r'[ \t]*$',
            re.MULTILINE | re.IGNORECASE
        )

        # Find numbered sections on same line (subsections)
        for match in numbered_same_line.finditer(text):
            number = match.group(1)
            title = match.group(2).strip()
            if should_skip(f"{number} {title}"):
                continue
            if is_toc_title(title):
                continue
            level = number.count('.') + 1
            headers.append({
                'number': number,
                'title': title,
                'level': level,
                'pos': match.start(),
                'page': page_num,
                'full_header': f"{number} {title}"
            })

        # Find top-level numbered sections with split lines
        for match in numbered_split_line.finditer(text):
            number = match.group(1)
            title_full = match.group(2).strip()
            if should_skip(f"{number} {title_full}"):
                continue
            if is_toc_title(title_full):
                continue
            # Skip if number is too large (likely a year or other number)
            if int(number) > 50:
                continue
            # Truncate title to reasonable length for storage
            title = title_full[:80].strip()
            headers.append({
                'number': number,
                'title': title,
                'level': 1,
                'pos': match.start(),
                'page': page_num,
                'full_header': f"{number} {title}"
            })

        # Find subsections with split lines
        for match in subsection_split_line.finditer(text):
            number = match.group(1)
            title_full = match.group(2).strip()
            if should_skip(f"{number} {title_full}"):
                continue
            if is_toc_title(title_full):
                continue
            # Truncate title to reasonable length for storage
            title = title_full[:80].strip()
            level = number.count('.') + 1
            headers.append({
                'number': number,
                'title': title,
                'level': level,
                'pos': match.start(),
                'page': page_num,
                'full_header': f"{number} {title}"
            })

        # Find named sections (Section 1, Chapter 2, etc.)
        for match in named_pattern.finditer(text):
            number = match.group(1).strip()
            title = (match.group(2) or "").strip()
            if should_skip(f"{number} {title}"):
                continue
            level = 1
            if '.' in number:
                level = number.count('.') + 1
            headers.append({
                'number': number,
                'title': title,
                'level': level,
                'pos': match.start(),
                'page': page_num,
                'full_header': f"{number}: {title}" if title else number
            })

        # Remove duplicates (same position)
        seen_positions = set()
        unique_headers = []
        for h in headers:
            if h['pos'] not in seen_positions:
                seen_positions.add(h['pos'])
                unique_headers.append(h)

        # Sort by position
        unique_headers.sort(key=lambda x: x['pos'])

        return unique_headers

    def _build_sections_from_pages(self, pages: List[Dict], min_section_chars: int = 200) -> List[Dict]:
        """
        Build section-based chunks from page data.

        Each section includes:
        - Section number/title
        - Start/end page numbers
        - Full text content
        - Parent section path (for hierarchy)

        Args:
            pages: List of page dicts from _extract_pages
            min_section_chars: Minimum characters for a section to be standalone

        Returns:
            List of section dicts with text, page ranges, and hierarchy info
        """
        # Create page lookup by page number
        page_by_num = {p['page']: p for p in pages}

        # First pass: collect all headers across all pages with their positions
        all_headers = []
        for page in pages:
            page_headers = self._detect_section_headers(page['text'], page['page'])
            for h in page_headers:
                h['page_text'] = page['text']  # Keep reference to page text
            all_headers.extend(page_headers)

        # Sort by page number, then by position within page
        all_headers.sort(key=lambda x: (x['page'], x['pos']))

        # If no headers found, fall back to page-based chunking
        if not all_headers:
            return self._fallback_page_chunks(pages)

        # Build sections by extracting text from start of each section
        sections = []
        section_stack = []  # Track current hierarchy

        for i, header in enumerate(all_headers):
            start_page = header['page']

            # Determine end page (page before next section, or last page)
            if i + 1 < len(all_headers):
                next_header = all_headers[i + 1]
                if next_header['page'] == start_page:
                    # Next section on same page - extract only up to next section
                    end_page = start_page
                    # Extract text from this header to next header on same page
                    page_text = header['page_text']
                    start_pos = header['pos']
                    end_pos = next_header['pos']
                    section_text = page_text[start_pos:end_pos].strip()
                else:
                    # Next section on different page
                    end_page = next_header['page'] - 1
                    # Extract text: from header pos to end of start_page,
                    # plus full text of pages between, up to (but not including) end_page+1
                    parts = []
                    # First page: from header position to end
                    page_text = header['page_text']
                    parts.append(page_text[header['pos']:].strip())
                    # Middle pages (if any)
                    for p in range(start_page + 1, end_page + 1):
                        if p in page_by_num:
                            parts.append(page_by_num[p]['text'].strip())
                    section_text = '\n\n'.join(parts)
            else:
                # Last section - goes to end of document
                end_page = pages[-1]['page']
                parts = []
                page_text = header['page_text']
                parts.append(page_text[header['pos']:].strip())
                for p in range(start_page + 1, end_page + 1):
                    if p in page_by_num:
                        parts.append(page_by_num[p]['text'].strip())
                section_text = '\n\n'.join(parts)

            # Skip very short sections (likely just headers)
            if len(section_text) < min_section_chars:
                continue

            # Update section stack for hierarchy
            level = header['level']
            while section_stack and section_stack[-1]['level'] >= level:
                section_stack.pop()
            section_stack.append(header)

            # Build section path
            section_path = ' > '.join(h['full_header'] for h in section_stack)

            sections.append({
                'number': header['number'],
                'title': header['title'],
                'full_header': header['full_header'],
                'level': header['level'],
                'section_path': section_path,
                'text': section_text,
                'start_page': start_page,
                'end_page': end_page,
                'char_count': len(section_text)
            })

        return sections

    def _fallback_page_chunks(self, pages: List[Dict], pages_per_chunk: int = 3) -> List[Dict]:
        """
        Fallback chunking when no sections detected - chunk by page groups.

        Args:
            pages: List of page dicts
            pages_per_chunk: Number of pages per chunk

        Returns:
            List of section-like dicts
        """
        chunks = []
        for i in range(0, len(pages), pages_per_chunk):
            page_group = pages[i:i + pages_per_chunk]
            text = "\n".join(p['text'] for p in page_group)
            start_page = page_group[0]['page']
            end_page = page_group[-1]['page']

            chunks.append({
                'number': '',
                'title': f'Pages {start_page}-{end_page}',
                'full_header': f'Pages {start_page}-{end_page}',
                'level': 1,
                'section_path': '',
                'text': text,
                'start_page': start_page,
                'end_page': end_page,
                'char_count': len(text)
            })

        return chunks

    def _split_large_section(self, section: Dict, max_chars: int = 6000) -> List[Dict]:
        """
        Split a section that's too large into smaller chunks.

        Tries to split on paragraph boundaries while preserving section metadata.
        """
        if section['char_count'] <= max_chars:
            return [section]

        text = section['text']
        chunks = []

        # Split on double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = ""
        chunk_num = 1

        for para in paragraphs:
            if len(current_chunk) + len(para) <= max_chars:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append({
                        **section,
                        'title': f"{section['title']} (Part {chunk_num})" if section['title'] else f"Part {chunk_num}",
                        'full_header': f"{section['full_header']} (Part {chunk_num})",
                        'text': current_chunk,
                        'char_count': len(current_chunk)
                    })
                    chunk_num += 1
                current_chunk = para

        # Don't forget the last chunk
        if current_chunk:
            chunks.append({
                **section,
                'title': f"{section['title']} (Part {chunk_num})" if section['title'] else f"Part {chunk_num}",
                'full_header': f"{section['full_header']} (Part {chunk_num})",
                'text': current_chunk,
                'char_count': len(current_chunk)
            })

        return chunks

    def process_file_sectioned(self, file_path: str, url: Optional[str] = None,
                                max_section_chars: int = 6000,
                                min_section_chars: int = 200,
                                start_page: int = 1,
                                end_page: Optional[int] = None) -> List[ScrapedPage]:
        """
        Process a PDF with section-aware chunking and page number tracking.

        This is the recommended method for building codes, manuals, and other
        structured documents where section hierarchy and page citations matter.

        Features:
        - Detects section headers (1, 1.1, 1.1.1, Chapter 1, etc.)
        - Preserves section hierarchy in metadata
        - Tracks page numbers for each chunk
        - Generates #page=N URLs for citations
        - Splits large sections while preserving context

        Args:
            file_path: Path to the PDF file
            url: Optional URL if downloaded from web (used as base for page URLs)
            max_section_chars: Maximum characters per chunk (splits larger sections)
            min_section_chars: Minimum characters for a standalone section
            start_page: Page number to start processing from (skip TOC, title pages)
            end_page: Optional page number to stop at (exclude appendices, index, etc.)

        Returns:
            List of ScrapedPage objects, one per section/chunk
        """
        path = Path(file_path)
        if not path.exists():
            print(f"File not found: {file_path}")
            return []

        if not path.suffix.lower() == '.pdf':
            print(f"Not a PDF file: {file_path}")
            return []

        try:
            # Extract pages with page numbers
            all_pages = self._extract_pages(str(path))

            # Filter to requested page range (skip TOC, title pages, appendices, etc.)
            if end_page is None:
                end_page = len(all_pages)
            pages = [p for p in all_pages if start_page <= p['page'] <= end_page]

            if start_page > 1 or end_page < len(all_pages):
                print(f"Processing pages {start_page}-{end_page} of {len(all_pages)} total")

            if not pages or sum(p['char_count'] for p in pages) < 100:
                print(f"PDF has too little text: {file_path}")
                return []

            # Get metadata
            metadata = self._extract_metadata(str(path))
            base_title = metadata.get('title') or self._clean_filename_to_title(path.name)

            # Build base URL
            doc_url = url or f"file://{path.absolute()}"

            # Get categories from metadata
            categories = []
            if metadata.get('keywords'):
                categories = [k.strip() for k in metadata['keywords'].split(',')][:10]
            if metadata.get('subject'):
                categories.append(metadata['subject'])

            # Build sections from pages
            sections = self._build_sections_from_pages(pages, min_section_chars)

            print(f"Detected {len(sections)} sections in {path.name}")

            # Process sections into ScrapedPage objects
            results = []
            for section in sections:
                # Split large sections
                chunks = self._split_large_section(section, max_section_chars)

                for chunk in chunks:
                    # Build title with section info
                    if chunk['number']:
                        title = f"{base_title} - {chunk['number']} {chunk['title']}"
                    elif chunk['title']:
                        title = f"{base_title} - {chunk['title']}"
                    else:
                        title = f"{base_title} (Pages {chunk['start_page']}-{chunk['end_page']})"

                    # Build URL with page reference
                    # Use start page for the #page= fragment
                    chunk_url = f"{doc_url}#page={chunk['start_page']}"

                    # Build content with section path context
                    content = chunk['text']
                    if chunk['section_path'] and chunk['section_path'] != chunk['full_header']:
                        # Add breadcrumb at top for context
                        content = f"[{chunk['section_path']}]\n\n{content}"

                    page = ScrapedPage(
                        url=chunk_url,
                        title=title,
                        content=content,
                        source=self.source_name,
                        categories=categories[:10],
                        last_modified=parse_pdf_date(metadata.get("creation_date")),
                        content_hash=self._hash_content(content),
                        scraped_at=datetime.utcnow().isoformat()
                    )
                    results.append(page)

            print(f"Created {len(results)} chunks from {len(sections)} sections")
            return results

        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_pdf_info(self, file_path: str, start_page: int = 1,
                      end_page: Optional[int] = None) -> Dict:
        """
        Get summary info about a PDF without full processing.

        Useful for previewing a PDF before deciding how to process it.

        Args:
            file_path: Path to the PDF file
            start_page: Page to start analysis from (skip TOC, title pages)
            end_page: Optional page to stop at

        Returns:
            Dict with page_count, total_chars, detected_sections, metadata
        """
        path = Path(file_path)
        if not path.exists():
            return {'error': f'File not found: {file_path}'}

        try:
            all_pages = self._extract_pages(str(path))
            metadata = self._extract_metadata(str(path))

            # Filter to requested page range
            if end_page is None:
                end_page = len(all_pages)
            pages = [p for p in all_pages if start_page <= p['page'] <= end_page]

            # Count sections in filtered range
            all_headers = []
            for page in pages:
                headers = self._detect_section_headers(page['text'], page['page'])
                all_headers.extend(headers)

            return {
                'file_name': path.name,
                'total_pages': len(all_pages),
                'analyzed_pages': len(pages),
                'page_range': f"{start_page}-{end_page}",
                'total_chars': sum(p['char_count'] for p in pages),
                'detected_sections': len(all_headers),
                'section_preview': all_headers[:20],  # First 20 sections
                'metadata': metadata
            }

        except Exception as e:
            return {'error': str(e)}


def create_pdf_scraper(source_name: str = "pdf", **kwargs) -> PDFScraper:
    """Convenience function to create a PDF scraper"""
    return PDFScraper(source_name=source_name, **kwargs)


# Quick test
if __name__ == "__main__":
    print("PDF Scraper")
    print(f"  PyMuPDF available: {HAS_PYMUPDF}")
    print(f"  pypdf available: {HAS_PYPDF}")

    if not HAS_PYMUPDF and not HAS_PYPDF:
        print("\nNo PDF library installed. Install with:")
        print("  pip install pymupdf")
        print("  or")
        print("  pip install pypdf")
    else:
        print("\nPDF scraper ready to use!")
        print("\nExample usage:")
        print('  scraper = PDFScraper(source_name="research")')
        print('  doc = scraper.process_file("paper.pdf")')
        print('  docs = scraper.crawl_for_pdfs("https://example.com/papers/")')
