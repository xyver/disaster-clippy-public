"""
Indexer module for ingesting content from ZIM files and HTML backups into the vector database.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Callable

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectordb import VectorStore, MetadataIndex


class ZIMIndexer:
    """
    Indexes content from ZIM files into the vector database.
    Uses zimply-core library to read ZIM archives.
    """

    def __init__(self, zim_path: str, source_id: str):
        self.zim_path = Path(zim_path)
        self.source_id = source_id

        if not self.zim_path.exists():
            raise FileNotFoundError(f"ZIM file not found: {zim_path}")

    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove scripts, styles, navigation
        for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()

        # Get text
        text = soup.get_text(separator=' ', strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = ' '.join(lines)

        return text

    def _update_metadata_index(self, documents: List[Dict]):
        """Update the metadata index with indexed documents"""
        try:
            metadata_dir = Path(__file__).parent.parent / "data" / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)

            # Create source-specific metadata file
            source_docs = {}
            total_chars = 0

            for doc in documents:
                doc_id = doc.get("id", doc.get("content_hash", ""))
                char_count = doc.get("char_count", len(doc.get("content", "")))
                total_chars += char_count

                source_docs[doc_id] = {
                    "title": doc.get("title", "Unknown"),
                    "url": doc.get("url", ""),
                    "content_hash": doc.get("content_hash", ""),
                    "scraped_at": doc.get("scraped_at", datetime.now().isoformat()),
                    "char_count": char_count
                }

            # Write source metadata file
            source_meta = {
                "version": 2,
                "source_id": self.source_id,
                "last_updated": datetime.now().isoformat(),
                "document_count": len(source_docs),
                "total_chars": total_chars,
                "documents": source_docs
            }

            source_file = metadata_dir / f"{self.source_id}.json"
            with open(source_file, 'w', encoding='utf-8') as f:
                json.dump(source_meta, f, indent=2)

            # Update master file
            master_file = metadata_dir / "_master.json"
            if master_file.exists():
                with open(master_file, 'r', encoding='utf-8') as f:
                    master = json.load(f)
            else:
                master = {"version": 2, "sources": {}}

            master["sources"][self.source_id] = {
                "count": len(source_docs),
                "chars": total_chars,
                "last_sync": datetime.now().isoformat(),
                "file": f"{self.source_id}.json",
                "topics": []
            }

            # Recalculate totals
            master["total_documents"] = sum(s["count"] for s in master["sources"].values())
            master["total_chars"] = sum(s["chars"] for s in master["sources"].values())
            master["last_updated"] = datetime.now().isoformat()

            with open(master_file, 'w', encoding='utf-8') as f:
                json.dump(master, f, indent=2)

            print(f"Updated metadata index: {len(source_docs)} documents for {self.source_id}")

        except Exception as e:
            print(f"Warning: Failed to update metadata index: {e}")

    def _get_title_from_html(self, html_content: str, url: str) -> str:
        """Extract title from HTML or fallback to URL"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Try <title> tag
        title_tag = soup.find('title')
        if title_tag and title_tag.text.strip():
            return title_tag.text.strip()

        # Try <h1> tag
        h1_tag = soup.find('h1')
        if h1_tag and h1_tag.text.strip():
            return h1_tag.text.strip()

        # Fallback to URL
        return url.split('/')[-1].replace('_', ' ').replace('.html', '')

    def index(self, limit: int = 1000, progress_callback: Optional[Callable] = None) -> Dict:
        """
        Index content from ZIM file into vector database.

        Args:
            limit: Maximum number of articles to index
            progress_callback: Function(current, total, message) for progress updates

        Returns:
            Dict with success status, count, errors
        """
        try:
            from zimply_core.zim_core import ZIMFile
        except ImportError:
            return {
                "success": False,
                "error": "zimply-core not installed. Run: pip install zimply-core",
                "indexed_count": 0
            }

        errors = []
        documents = []

        try:
            if progress_callback:
                progress_callback(0, limit, "Opening ZIM file...")

            zim = ZIMFile(str(self.zim_path), 'utf-8')

            # Get article count from header
            article_count = zim.header_fields.get('articleCount', 0)
            print(f"ZIM file contains {article_count} articles")

            if progress_callback:
                progress_callback(0, min(limit, article_count), f"Found {article_count} articles in ZIM")

            # Iterate through articles by ID
            indexed = 0
            skipped = 0
            seen_ids = set()

            for i in range(article_count):
                if indexed >= limit:
                    break

                try:
                    article = zim.get_article_by_id(i)
                    if article is None:
                        skipped += 1
                        continue

                    # Check mimetype - only want HTML articles
                    mimetype = getattr(article, 'mimetype', '')
                    if 'text/html' not in str(mimetype).lower():
                        skipped += 1
                        continue

                    # Get content
                    content = article.data
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='ignore')

                    if not content or len(content) < 100:
                        skipped += 1
                        continue

                    # Get URL and title
                    url = getattr(article, 'url', '') or f"article_{i}"
                    title = getattr(article, 'title', '') or self._get_title_from_html(content, url)

                    # Skip special pages
                    if any(x in url.lower() for x in ['special:', 'file:', 'category:', 'template:', 'mediawiki:', '-/', 'favicon']):
                        skipped += 1
                        continue

                    # Extract text
                    text = self._extract_text_from_html(content)
                    if len(text) < 50:
                        skipped += 1
                        continue

                    # Create document
                    full_url = f"zim://{self.source_id}/{url}"
                    doc_id = hashlib.md5(f"{self.source_id}:{url}".encode()).hexdigest()

                    # Skip duplicates
                    if doc_id in seen_ids:
                        skipped += 1
                        continue
                    seen_ids.add(doc_id)

                    documents.append({
                        "id": doc_id,
                        "content": text[:50000],  # Limit content size
                        "title": title,
                        "url": full_url,
                        "source": self.source_id,
                        "categories": [],
                        "content_hash": hashlib.md5(text.encode()).hexdigest(),
                        "scraped_at": datetime.now().isoformat(),
                        "char_count": len(text),
                        "doc_type": "article",
                        "from_zim": True
                    })

                    indexed += 1

                    if progress_callback and indexed % 10 == 0:
                        progress_callback(indexed, min(limit, article_count),
                                        f"Extracted: {title[:50]}...")

                except Exception as e:
                    errors.append(f"Error processing article {i}: {str(e)}")
                    continue

            print(f"Extracted {len(documents)} articles from ZIM (skipped {skipped} non-articles)")

            if not documents:
                return {
                    "success": False,
                    "error": "No articles found in ZIM file",
                    "indexed_count": 0,
                    "errors": errors
                }

            # Add to vector store
            if progress_callback:
                progress_callback(len(documents), len(documents), "Computing embeddings and storing...")

            print("Adding documents to vector store...")
            store = VectorStore()
            count = store.add_documents(documents)

            # Update metadata index to track this source
            if count > 0:
                self._update_metadata_index(documents)

            return {
                "success": True,
                "indexed_count": count,
                "total_extracted": len(documents),
                "skipped": skipped,
                "errors": errors
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "indexed_count": 0,
                "errors": errors
            }


class HTMLBackupIndexer:
    """
    Indexes content from HTML backup folders into the vector database.
    Reads from the manifest.json and pages/ folder created by html_backup.py
    """

    def __init__(self, backup_path: str, source_id: str):
        self.backup_path = Path(backup_path)
        self.source_id = source_id
        self.manifest_path = self.backup_path / "manifest.json"
        self.pages_dir = self.backup_path / "pages"

        if not self.backup_path.exists():
            raise FileNotFoundError(f"Backup folder not found: {backup_path}")

    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove scripts, styles, navigation, ads
        for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
            tag.decompose()

        # Try to find main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('div', {'id': 'content'}) or soup.find('div', {'class': 'mw-parser-output'})

        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = ' '.join(lines)

        return text

    def index(self, limit: int = 1000, progress_callback: Optional[Callable] = None,
              skip_existing: bool = True) -> Dict:
        """
        Index content from HTML backup into vector database.

        Args:
            limit: Maximum number of pages to index
            progress_callback: Function(current, total, message) for progress updates
            skip_existing: Skip pages already in the database

        Returns:
            Dict with success status, count, errors
        """
        errors = []
        documents = []

        # Load manifest
        if not self.manifest_path.exists():
            return {
                "success": False,
                "error": "No manifest.json found in backup folder",
                "indexed_count": 0
            }

        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        pages = manifest.get("pages", {})
        if not pages:
            return {
                "success": False,
                "error": "No pages found in manifest",
                "indexed_count": 0
            }

        print(f"Found {len(pages)} pages in backup manifest")

        # Get existing URLs if skipping
        existing_urls = set()
        if skip_existing:
            store = VectorStore()
            existing_urls = store.metadata_index.get_urls(self.source_id)
            print(f"Found {len(existing_urls)} already indexed URLs")

        if progress_callback:
            progress_callback(0, min(limit, len(pages)), "Loading pages from backup...")

        # Process each page
        indexed = 0
        skipped = 0

        for url, page_info in pages.items():
            if indexed >= limit:
                break

            # Skip if already indexed
            if skip_existing and url in existing_urls:
                skipped += 1
                continue

            try:
                filename = page_info.get("filename")
                title = page_info.get("title", "Untitled")

                if not filename:
                    errors.append(f"No filename for {url}")
                    continue

                # Read HTML file
                html_path = self.pages_dir / filename
                if not html_path.exists():
                    errors.append(f"File not found: {filename}")
                    continue

                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # Extract text
                text = self._extract_text_from_html(html_content)
                if len(text) < 50:
                    skipped += 1
                    continue

                # Create document
                doc_id = hashlib.md5(f"{self.source_id}:{url}".encode()).hexdigest()

                documents.append({
                    "id": doc_id,
                    "content": text[:50000],  # Limit content size
                    "title": title,
                    "url": url,
                    "source": self.source_id,
                    "categories": [],
                    "content_hash": hashlib.md5(text.encode()).hexdigest(),
                    "scraped_at": datetime.now().isoformat(),
                    "char_count": len(text),
                    "doc_type": "article",
                    "from_backup": True
                })

                indexed += 1

                if progress_callback and indexed % 10 == 0:
                    progress_callback(indexed, min(limit, len(pages) - skipped),
                                    f"Processing: {title[:50]}...")

            except Exception as e:
                errors.append(f"Error processing {url}: {str(e)}")
                continue

        print(f"Prepared {len(documents)} documents for indexing (skipped {skipped})")

        if not documents:
            if skipped > 0:
                return {
                    "success": True,
                    "indexed_count": 0,
                    "skipped": skipped,
                    "message": "All pages already indexed",
                    "errors": errors
                }
            return {
                "success": False,
                "error": "No documents to index",
                "indexed_count": 0,
                "errors": errors
            }

        # Add to vector store
        if progress_callback:
            progress_callback(len(documents), len(documents), "Computing embeddings and storing...")

        print("Adding documents to vector store...")
        store = VectorStore()
        count = store.add_documents(documents)

        # Update metadata index to track this source
        if count > 0:
            self._update_metadata_index(documents)

        return {
            "success": True,
            "indexed_count": count,
            "total_processed": len(documents),
            "skipped": skipped,
            "errors": errors
        }

    def _update_metadata_index(self, documents: List[Dict]):
        """Update the metadata index with indexed documents"""
        try:
            metadata_dir = Path(__file__).parent.parent / "data" / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)

            # Load existing source metadata or create new
            source_file = metadata_dir / f"{self.source_id}.json"
            if source_file.exists():
                with open(source_file, 'r', encoding='utf-8') as f:
                    source_meta = json.load(f)
                source_docs = source_meta.get("documents", {})
            else:
                source_docs = {}

            # Add new documents
            total_chars = sum(d.get("char_count", 0) for d in source_docs.values()) if isinstance(source_docs, dict) else 0

            for doc in documents:
                doc_id = doc.get("id", doc.get("content_hash", ""))
                char_count = doc.get("char_count", len(doc.get("content", "")))
                total_chars += char_count

                source_docs[doc_id] = {
                    "title": doc.get("title", "Unknown"),
                    "url": doc.get("url", ""),
                    "content_hash": doc.get("content_hash", ""),
                    "scraped_at": doc.get("scraped_at", datetime.now().isoformat()),
                    "char_count": char_count
                }

            # Write source metadata file
            source_meta = {
                "version": 2,
                "source_id": self.source_id,
                "last_updated": datetime.now().isoformat(),
                "document_count": len(source_docs),
                "total_chars": total_chars,
                "documents": source_docs
            }

            with open(source_file, 'w', encoding='utf-8') as f:
                json.dump(source_meta, f, indent=2)

            # Update master file
            master_file = metadata_dir / "_master.json"
            if master_file.exists():
                with open(master_file, 'r', encoding='utf-8') as f:
                    master = json.load(f)
            else:
                master = {"version": 2, "sources": {}}

            master["sources"][self.source_id] = {
                "count": len(source_docs),
                "chars": total_chars,
                "last_sync": datetime.now().isoformat(),
                "file": f"{self.source_id}.json",
                "topics": []
            }

            # Recalculate totals
            master["total_documents"] = sum(s["count"] for s in master["sources"].values())
            master["total_chars"] = sum(s["chars"] for s in master["sources"].values())
            master["last_updated"] = datetime.now().isoformat()

            with open(master_file, 'w', encoding='utf-8') as f:
                json.dump(master, f, indent=2)

            print(f"Updated metadata index: {len(source_docs)} documents for {self.source_id}")

        except Exception as e:
            print(f"Warning: Failed to update metadata index: {e}")


def index_zim_file(zim_path: str, source_id: str, limit: int = 1000,
                   progress_callback: Optional[Callable] = None) -> Dict:
    """
    Convenience function to index a ZIM file.

    Args:
        zim_path: Path to ZIM file
        source_id: Source identifier for the content
        limit: Maximum articles to index
        progress_callback: Progress callback function

    Returns:
        Dict with results
    """
    indexer = ZIMIndexer(zim_path, source_id)
    return indexer.index(limit=limit, progress_callback=progress_callback)


def index_html_backup(backup_path: str, source_id: str, limit: int = 1000,
                      progress_callback: Optional[Callable] = None,
                      skip_existing: bool = True) -> Dict:
    """
    Convenience function to index an HTML backup.

    Args:
        backup_path: Path to backup folder (containing manifest.json)
        source_id: Source identifier for the content
        limit: Maximum pages to index
        progress_callback: Progress callback function
        skip_existing: Skip pages already in database

    Returns:
        Dict with results
    """
    indexer = HTMLBackupIndexer(backup_path, source_id)
    return indexer.index(limit=limit, progress_callback=progress_callback,
                        skip_existing=skip_existing)


if __name__ == "__main__":
    # Test with command line args
    import sys

    if len(sys.argv) < 3:
        print("Usage: python indexer.py <zim|html> <path> [source_id] [limit]")
        print("  python indexer.py zim D:\\backups\\bitcoin.zim bitcoin 100")
        print("  python indexer.py html D:\\backups\\solarcooking solarcooking 100")
        sys.exit(1)

    index_type = sys.argv[1]
    path = sys.argv[2]
    source_id = sys.argv[3] if len(sys.argv) > 3 else Path(path).stem
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    def progress(current, total, message):
        print(f"[{current}/{total}] {message}")

    if index_type == "zim":
        result = index_zim_file(path, source_id, limit, progress)
    elif index_type == "html":
        result = index_html_backup(path, source_id, limit, progress)
    else:
        print(f"Unknown type: {index_type}")
        sys.exit(1)

    print("\nResult:")
    print(json.dumps(result, indent=2))
