"""
Indexer module for ingesting content from ZIM files and HTML backups into the vector database.

Outputs v2 schema format:
    - {source_id}_source.json (source-level metadata)
    - {source_id}_documents.json (document metadata)
    - {source_id}_embeddings.json (vectors only, no content duplication)
    - {source_id}_manifest.json (distribution manifest)
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
from .schemas import (
    get_source_file, get_documents_file, get_embeddings_file,
    get_distribution_manifest_file, CURRENT_SCHEMA_VERSION
)


# =============================================================================
# SHARED V2 OUTPUT FUNCTIONS
# =============================================================================

def save_source_metadata(output_folder: Path, source_id: str, documents: List[Dict],
                         base_url: str = "", source_type: str = "unknown",
                         license_info: str = "Unknown") -> Optional[Path]:
    """
    Save source-level metadata file ({source_id}_source.json).

    This is the top-level summary of the source for quick browsing.
    Preserves user-edited fields (name, description, license, license_verified, etc.)
    if the file already exists.
    """
    try:
        total_chars = sum(doc.get("char_count", len(doc.get("content", ""))) for doc in documents)

        # Build category stats
        categories = {}
        for doc in documents:
            cats = doc.get("categories", [])
            if isinstance(cats, str):
                try:
                    cats = json.loads(cats)
                except:
                    cats = []
            for cat in cats:
                if cat:
                    if cat not in categories:
                        categories[cat] = {"count": 0, "total_chars": 0}
                    categories[cat]["count"] += 1
                    categories[cat]["total_chars"] += doc.get("char_count", 0)

        source_file = output_folder / get_source_file(source_id)

        # Load existing source config to preserve user-edited fields
        existing_meta = {}
        if source_file.exists():
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    existing_meta = json.load(f)
            except Exception:
                pass

        # User-edited fields that should be preserved
        preserved_fields = ["name", "description", "license", "license_verified",
                           "base_url", "tags", "created_at"]

        source_meta = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "source_id": source_id,
            "name": source_id.replace('_', ' ').title(),
            "description": "",
            "license": license_info,
            "license_verified": False,
            "base_url": base_url,
            "total_docs": len(documents),
            "total_chars": total_chars,
            "categories": categories,
            "created_at": existing_meta.get("created_at", datetime.now().isoformat()),
            "last_backup": existing_meta.get("last_backup", ""),
            "last_indexed": datetime.now().isoformat()
        }

        # Preserve user-edited fields from existing file
        for field in preserved_fields:
            if field in existing_meta and existing_meta[field]:
                source_meta[field] = existing_meta[field]

        with open(source_file, 'w', encoding='utf-8') as f:
            json.dump(source_meta, f, indent=2, ensure_ascii=False)

        print(f"Saved source metadata: {source_file}")
        return source_file

    except Exception as e:
        print(f"Warning: Failed to save source metadata: {e}")
        return None


def save_documents_metadata(output_folder: Path, source_id: str,
                            documents: List[Dict]) -> Optional[Path]:
    """
    Save document metadata file ({source_id}_documents.json).

    This contains per-document metadata for quick scanning without loading embeddings.
    """
    try:
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
                "char_count": char_count,
                "categories": doc.get("categories", [])
            }

        docs_data = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "source_id": source_id,
            "document_count": len(source_docs),
            "total_chars": total_chars,
            "last_updated": datetime.now().isoformat(),
            "documents": source_docs
        }

        docs_file = output_folder / get_documents_file(source_id)
        with open(docs_file, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)

        print(f"Saved documents metadata: {docs_file} ({len(source_docs)} documents)")
        return docs_file

    except Exception as e:
        print(f"Warning: Failed to save documents metadata: {e}")
        return None


def save_embeddings_file(output_folder: Path, source_id: str,
                         index_data: Dict) -> Optional[Path]:
    """
    Save embeddings file ({source_id}_embeddings.json).

    V2 format: vectors only, no content duplication.
    """
    try:
        ids = index_data.get("ids", [])
        embeddings = index_data.get("embeddings", [])

        vectors = {}
        for i in range(len(ids)):
            if i < len(embeddings) and embeddings[i]:
                vectors[ids[i]] = embeddings[i]

        embeddings_data = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "source_id": source_id,
            "embedding_model": "text-embedding-3-small",
            "dimensions": 1536,
            "document_count": len(vectors),
            "created_at": datetime.now().isoformat(),
            "vectors": vectors
        }

        embeddings_file = output_folder / get_embeddings_file(source_id)
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f)  # No indent - keep compact

        print(f"Saved embeddings: {embeddings_file} ({len(vectors)} vectors)")
        return embeddings_file

    except Exception as e:
        print(f"Warning: Failed to save embeddings file: {e}")
        return None


def save_distribution_manifest(output_folder: Path, source_id: str, documents: List[Dict],
                               source_type: str = "unknown", backup_info: Dict = None,
                               license_info: str = "Unknown", base_url: str = "") -> Optional[Path]:
    """
    Save distribution manifest file ({source_id}_manifest.json).

    Contains package info for R2 distribution.
    """
    try:
        total_chars = sum(doc.get("char_count", len(doc.get("content", ""))) for doc in documents)

        manifest = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "source_id": source_id,
            "name": source_id.replace('_', ' ').title(),
            "description": "",
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "document_count": len(documents),
            "has_embeddings": True,
            "has_categories": False,
            "license": license_info,
            "attribution": "",
            "base_url": base_url,
            "source_type": source_type,
            "backup_info": backup_info or {},
            "files": [
                get_source_file(source_id),
                get_documents_file(source_id),
                get_embeddings_file(source_id)
            ],
            "total_chars": total_chars
        }

        manifest_file = output_folder / get_distribution_manifest_file(source_id)
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Saved distribution manifest: {manifest_file}")
        return manifest_file

    except Exception as e:
        print(f"Warning: Failed to save distribution manifest: {e}")
        return None


class ZIMIndexer:
    """
    Indexes content from ZIM files into the vector database.
    Uses zimply-core library to read ZIM archives.
    """

    def __init__(self, zim_path: str, source_id: str, backup_folder: str = None):
        self.zim_path = Path(zim_path)
        self.source_id = source_id
        # Output folder: use backup_folder directly (caller should pass the source folder)
        if backup_folder:
            self.output_folder = Path(backup_folder)
        else:
            # Fallback to parent folder of ZIM file
            self.output_folder = self.zim_path.parent
        self.output_folder.mkdir(parents=True, exist_ok=True)

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

    def _save_metadata_file(self, documents: List[Dict]) -> Optional[Path]:
        """Save the per-document metadata file to output folder"""
        try:
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

            # Write source metadata file to output folder
            source_meta = {
                "version": 2,
                "source_id": self.source_id,
                "last_updated": datetime.now().isoformat(),
                "document_count": len(source_docs),
                "total_chars": total_chars,
                "documents": source_docs
            }

            metadata_file = self.output_folder / f"{self.source_id}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(source_meta, f, indent=2)

            print(f"Saved metadata file: {metadata_file} ({len(source_docs)} documents)")

            # Update master metadata file
            try:
                from sourcepacks.pack_tools import update_master_metadata
                update_master_metadata(self.source_id, source_meta)
                print(f"Updated master metadata with {self.source_id}")
            except Exception as e:
                print(f"Warning: Failed to update master metadata: {e}")

            return metadata_file

        except Exception as e:
            print(f"Warning: Failed to save metadata file: {e}")
            return None

    def _save_index_file(self, index_data: Dict) -> Optional[Path]:
        """Save the vector index data to a JSON file for distribution"""
        try:
            index_file = self.output_folder / f"{self.source_id}_index.json"

            # Structure the index file
            index_export = {
                "version": 1,
                "source_id": self.source_id,
                "created_at": datetime.now().isoformat(),
                "document_count": len(index_data.get("ids", [])),
                "embedding_model": "default",
                "documents": []
            }

            # Build document entries with embeddings
            ids = index_data.get("ids", [])
            embeddings = index_data.get("embeddings", [])
            contents = index_data.get("contents", [])
            metadatas = index_data.get("metadatas", [])

            for i in range(len(ids)):
                doc_entry = {
                    "id": ids[i],
                    "embedding": embeddings[i] if i < len(embeddings) else None,
                    "content": contents[i] if i < len(contents) else None,
                    "metadata": metadatas[i] if i < len(metadatas) else {}
                }
                index_export["documents"].append(doc_entry)

            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_export, f)

            print(f"Saved index file: {index_file} ({len(ids)} documents)")
            return index_file

        except Exception as e:
            print(f"Warning: Failed to save index file: {e}")
            return None

    def _save_manifest_file(self, documents: List[Dict], backup_info: Dict = None) -> Optional[Path]:
        """Save the source-level manifest file"""
        try:
            manifest_file = self.output_folder / f"{self.source_id}_manifest.json"

            total_chars = sum(doc.get("char_count", len(doc.get("content", ""))) for doc in documents)

            manifest = {
                "version": 1,
                "source_id": self.source_id,
                "created_at": datetime.now().isoformat(),
                "document_count": len(documents),
                "total_chars": total_chars,
                "source_type": "zim",
                "backup_info": backup_info or {
                    "type": "zim",
                    "path": str(self.zim_path),
                    "size_mb": round(self.zim_path.stat().st_size / (1024*1024), 2) if self.zim_path.exists() else 0
                },
                "files": {
                    "index": f"{self.source_id}_index.json",
                    "metadata": f"{self.source_id}_metadata.json",
                    "manifest": f"{self.source_id}_manifest.json"
                }
            }

            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)

            print(f"Saved manifest file: {manifest_file}")
            return manifest_file

        except Exception as e:
            print(f"Warning: Failed to save manifest file: {e}")
            return None

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

            # Close ZIM file to release the handle
            zim.close()

            print(f"Extracted {len(documents)} articles from ZIM (skipped {skipped} non-articles)")

            if not documents:
                return {
                    "success": False,
                    "error": "No articles found in ZIM file",
                    "indexed_count": 0,
                    "errors": errors
                }

            # Add to vector store and get index data for saving
            if progress_callback:
                progress_callback(len(documents), len(documents), "Computing embeddings and storing...")

            print("Adding documents to vector store...")
            store = VectorStore()
            result = store.add_documents(documents, return_index_data=True)
            count = result["count"]
            index_data = result["index_data"]

            # Save all output files to backup folder (v2 format)
            if count > 0:
                # V2 format: separate files for source, documents, embeddings
                save_source_metadata(self.output_folder, self.source_id, documents,
                                    source_type="zim")
                save_documents_metadata(self.output_folder, self.source_id, documents)
                if index_data:
                    save_embeddings_file(self.output_folder, self.source_id, index_data)
                save_distribution_manifest(self.output_folder, self.source_id, documents,
                                          source_type="zim",
                                          backup_info={
                                              "type": "zim",
                                              "path": str(self.zim_path),
                                              "size_mb": round(self.zim_path.stat().st_size / (1024*1024), 2)
                                          })

                # Also save legacy format for backwards compatibility
                self._save_metadata_file(documents)

            return {
                "success": True,
                "indexed_count": count,
                "total_extracted": len(documents),
                "skipped": skipped,
                "errors": errors,
                "output_folder": str(self.output_folder)
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
    Reads from the {source_id}_backup_manifest.json and pages/ folder created by html_backup.py
    """

    def __init__(self, backup_path: str, source_id: str, backup_folder: str = None):
        self.backup_path = Path(backup_path)
        self.source_id = source_id
        # Try multiple manifest naming conventions
        self.manifest_path = None
        manifest_candidates = [
            self.backup_path / f"{source_id}_backup_manifest.json",  # New naming
            self.backup_path / f"{source_id}_manifest.json",         # V1 naming from R2
            self.backup_path / "manifest.json",                       # Legacy fallback
        ]
        for candidate in manifest_candidates:
            if candidate.exists():
                self.manifest_path = candidate
                break
        # Default to first option if none exist (will error later with clear message)
        if not self.manifest_path:
            self.manifest_path = manifest_candidates[0]
        self.pages_dir = self.backup_path / "pages"
        # Output folder: use backup_folder directly (caller should pass the source folder)
        if backup_folder:
            self.output_folder = Path(backup_folder)
        else:
            # For HTML backups, output to the backup path itself
            self.output_folder = self.backup_path
        self.output_folder.mkdir(parents=True, exist_ok=True)

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

        # Load manifest or scan pages folder
        pages = {}

        if self.manifest_path and self.manifest_path.exists():
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            pages = manifest.get("pages", {})

        # If no pages in manifest, scan the pages folder directly
        if not pages and self.pages_dir.exists():
            print(f"No pages in manifest, scanning {self.pages_dir}...")
            for html_file in self.pages_dir.glob("*.html"):
                # Create a synthetic page entry
                filename = html_file.name
                # Try to reconstruct URL from filename (reverse of html_backup.py naming)
                url_path = filename.replace(".html", "").replace("_", "/")
                url = f"/{url_path}"
                pages[url] = {
                    "filename": filename,
                    "title": filename.replace(".html", "").replace("_", " ")
                }
            print(f"Found {len(pages)} HTML files in pages folder")

        if not pages:
            return {
                "success": False,
                "error": "No pages found in manifest or pages folder",
                "indexed_count": 0
            }

        print(f"Found {len(pages)} pages to index")

        # Get existing document IDs if skipping
        existing_ids = set()
        if skip_existing:
            store = VectorStore()
            existing_ids = store.get_existing_ids()
            print(f"Found {len(existing_ids)} already indexed documents")

        if progress_callback:
            progress_callback(0, min(limit, len(pages)), "Loading pages from backup...")

        # Process each page
        indexed = 0
        skipped = 0

        for url, page_info in pages.items():
            if indexed >= limit:
                break

            # Generate doc_id from source and URL (consistent with how it was created)
            doc_id = hashlib.md5(f"{self.source_id}:{url}".encode()).hexdigest()

            # Skip if already indexed
            if skip_existing and doc_id in existing_ids:
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

        # Add to vector store and get index data for saving
        if progress_callback:
            progress_callback(len(documents), len(documents), "Computing embeddings and storing...")

        print("Adding documents to vector store...")
        store = VectorStore()
        result = store.add_documents(documents, return_index_data=True)
        count = result["count"]
        index_data = result["index_data"]

        # Save all output files to backup folder (v2 format)
        if count > 0:
            # Get base_url from manifest if available
            base_url = ""
            try:
                if self.manifest_path.exists():
                    with open(self.manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    base_url = manifest.get("base_url", "")
            except Exception:
                pass

            # V2 format: separate files for source, documents, embeddings
            save_source_metadata(self.output_folder, self.source_id, documents,
                                base_url=base_url, source_type="html")
            save_documents_metadata(self.output_folder, self.source_id, documents)
            if index_data:
                save_embeddings_file(self.output_folder, self.source_id, index_data)
            save_distribution_manifest(self.output_folder, self.source_id, documents,
                                       source_type="html", base_url=base_url)

            # Also save legacy format for backwards compatibility (can remove later)
            self._save_metadata_file(documents)

        return {
            "success": True,
            "indexed_count": count,
            "total_processed": len(documents),
            "skipped": skipped,
            "errors": errors,
            "output_folder": str(self.output_folder)
        }

    def _save_metadata_file(self, documents: List[Dict]) -> Optional[Path]:
        """Save the per-document metadata file to output folder"""
        try:
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

            # Write source metadata file to output folder
            source_meta = {
                "version": 2,
                "source_id": self.source_id,
                "last_updated": datetime.now().isoformat(),
                "document_count": len(source_docs),
                "total_chars": total_chars,
                "documents": source_docs
            }

            metadata_file = self.output_folder / f"{self.source_id}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(source_meta, f, indent=2)

            print(f"Saved metadata file: {metadata_file} ({len(source_docs)} documents)")

            # Update master metadata file
            try:
                from sourcepacks.pack_tools import update_master_metadata
                update_master_metadata(self.source_id, source_meta)
                print(f"Updated master metadata with {self.source_id}")
            except Exception as e:
                print(f"Warning: Failed to update master metadata: {e}")

            return metadata_file

        except Exception as e:
            print(f"Warning: Failed to save metadata file: {e}")
            return None

    def _save_index_file(self, index_data: Dict) -> Optional[Path]:
        """Save the vector index data to a JSON file for distribution"""
        try:
            index_file = self.output_folder / f"{self.source_id}_index.json"

            # Structure the index file
            index_export = {
                "version": 1,
                "source_id": self.source_id,
                "created_at": datetime.now().isoformat(),
                "document_count": len(index_data.get("ids", [])),
                "embedding_model": "default",
                "documents": []
            }

            # Build document entries with embeddings
            ids = index_data.get("ids", [])
            embeddings = index_data.get("embeddings", [])
            contents = index_data.get("contents", [])
            metadatas = index_data.get("metadatas", [])

            for i in range(len(ids)):
                doc_entry = {
                    "id": ids[i],
                    "embedding": embeddings[i] if i < len(embeddings) else None,
                    "content": contents[i] if i < len(contents) else None,
                    "metadata": metadatas[i] if i < len(metadatas) else {}
                }
                index_export["documents"].append(doc_entry)

            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_export, f)

            print(f"Saved index file: {index_file} ({len(ids)} documents)")
            return index_file

        except Exception as e:
            print(f"Warning: Failed to save index file: {e}")
            return None

    def _save_manifest_file(self, documents: List[Dict], backup_info: Dict = None) -> Optional[Path]:
        """Save the source-level manifest file"""
        try:
            manifest_file = self.output_folder / f"{self.source_id}_manifest.json"

            total_chars = sum(doc.get("char_count", len(doc.get("content", ""))) for doc in documents)

            # Calculate backup size if available
            backup_size_mb = 0
            if self.backup_path.exists():
                try:
                    total_size = sum(f.stat().st_size for f in self.backup_path.rglob('*') if f.is_file())
                    backup_size_mb = round(total_size / (1024*1024), 2)
                except Exception:
                    pass

            manifest = {
                "version": 1,
                "source_id": self.source_id,
                "created_at": datetime.now().isoformat(),
                "document_count": len(documents),
                "total_chars": total_chars,
                "source_type": "html",
                "backup_info": backup_info or {
                    "type": "html",
                    "path": str(self.backup_path),
                    "size_mb": backup_size_mb
                },
                "files": {
                    "index": f"{self.source_id}_index.json",
                    "metadata": f"{self.source_id}_metadata.json",
                    "manifest": f"{self.source_id}_manifest.json"
                }
            }

            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)

            print(f"Saved manifest file: {manifest_file}")
            return manifest_file

        except Exception as e:
            print(f"Warning: Failed to save manifest file: {e}")
            return None


class PDFIndexer:
    """
    Indexes content from PDF files or PDF collection folders into the vector database.
    Handles both individual PDFs and collections managed by PDFCollectionManager.
    """

    def __init__(self, source_path: str, source_id: str, backup_folder: str = None):
        """
        Args:
            source_path: Path to a PDF file or folder containing PDFs
            source_id: Source identifier for this content
            backup_folder: Optional folder to save output files
        """
        self.source_path = Path(source_path)
        self.source_id = source_id

        if backup_folder:
            self.output_folder = Path(backup_folder)
        else:
            # Default to source_path parent if it's a file, or source_path if folder
            if self.source_path.is_file():
                self.output_folder = self.source_path.parent
            else:
                self.output_folder = self.source_path
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Try to import PDF extraction library
        self.has_pymupdf = False
        self.has_pypdf = False
        try:
            import fitz
            self.has_pymupdf = True
        except ImportError:
            pass
        try:
            from pypdf import PdfReader
            self.has_pypdf = True
        except ImportError:
            pass

        if not self.has_pymupdf and not self.has_pypdf:
            raise ImportError(
                "No PDF library available. Install one of:\n"
                "  pip install pymupdf  (recommended)\n"
                "  pip install pypdf"
            )

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract text from a PDF file"""
        if self.has_pymupdf:
            import fitz
            doc = fitz.open(str(pdf_path))
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)
        elif self.has_pypdf:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n".join(text_parts)
        return ""

    def _extract_metadata(self, pdf_path: Path) -> dict:
        """Extract metadata from a PDF file"""
        try:
            if self.has_pymupdf:
                import fitz
                doc = fitz.open(str(pdf_path))
                metadata = doc.metadata or {}
                doc.close()
                return {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "keywords": metadata.get("keywords", ""),
                    "creator": metadata.get("creator", ""),
                }
            elif self.has_pypdf:
                from pypdf import PdfReader
                reader = PdfReader(str(pdf_path))
                if reader.metadata:
                    return {
                        "title": reader.metadata.get("/Title", ""),
                        "author": reader.metadata.get("/Author", ""),
                        "subject": reader.metadata.get("/Subject", ""),
                        "keywords": reader.metadata.get("/Keywords", ""),
                        "creator": reader.metadata.get("/Creator", ""),
                    }
        except Exception as e:
            print(f"Error extracting metadata from {pdf_path}: {e}")
        return {}

    def _clean_filename_to_title(self, filename: str) -> str:
        """Convert filename to readable title"""
        import re
        name = Path(filename).stem
        name = re.sub(r'[-_]+', ' ', name)
        name = re.sub(r'^\d{4}[-_]?\d{0,2}[-_]?\d{0,2}[-_]?', '', name)
        return name.strip().title()

    def _get_pdf_files(self) -> list:
        """Get list of PDF files to process"""
        if self.source_path.is_file():
            if self.source_path.suffix.lower() == '.pdf':
                return [self.source_path]
            return []
        elif self.source_path.is_dir():
            return list(self.source_path.glob("*.pdf"))
        return []

    def index(self, limit: int = 1000, progress_callback: Optional[Callable] = None,
              skip_existing: bool = True, chunk_size: int = 4000) -> Dict:
        """
        Index PDF content into vector database.

        Args:
            limit: Maximum number of PDFs to index
            progress_callback: Function(current, total, message) for progress updates
            skip_existing: Skip PDFs already in the database
            chunk_size: Characters per chunk for long PDFs

        Returns:
            Dict with success status, count, errors
        """
        errors = []
        documents = []
        skipped = 0

        pdf_files = self._get_pdf_files()[:limit]
        total_pdfs = len(pdf_files)

        if total_pdfs == 0:
            return {
                "success": False,
                "error": "No PDF files found",
                "indexed_count": 0
            }

        # Get existing IDs to check for duplicates
        existing_ids = set()
        if skip_existing:
            try:
                from vectordb.store import VectorStore
                store = VectorStore()
                existing_ids = store.get_existing_ids()
            except:
                pass

        for i, pdf_path in enumerate(pdf_files):
            if progress_callback:
                progress_callback(i + 1, total_pdfs, f"Processing {pdf_path.name}")

            try:
                # Extract text and metadata
                text = self._extract_text(pdf_path)
                if len(text) < 100:
                    errors.append(f"{pdf_path.name}: Too little text extracted")
                    continue

                metadata = self._extract_metadata(pdf_path)
                title = metadata.get("title") or self._clean_filename_to_title(pdf_path.name)

                # Generate content hash
                content_hash = hashlib.md5(text.encode()).hexdigest()[:12]
                doc_id = f"{self.source_id}_{content_hash}"

                if doc_id in existing_ids:
                    skipped += 1
                    continue

                # Chunk long PDFs
                if len(text) > chunk_size:
                    chunks = [text[j:j+chunk_size] for j in range(0, len(text), chunk_size)]
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_id = f"{doc_id}_chunk{chunk_idx}"
                        if chunk_id in existing_ids:
                            continue
                        documents.append({
                            "id": chunk_id,
                            "content": chunk,
                            "url": f"file://{pdf_path}#chunk{chunk_idx}",
                            "title": f"{title} (Part {chunk_idx + 1}/{len(chunks)})",
                            "source": self.source_id,
                            "categories": [],
                            "content_hash": f"{content_hash}_{chunk_idx}",
                            "char_count": len(chunk),
                            "pdf_file": pdf_path.name,
                        })
                else:
                    documents.append({
                        "id": doc_id,
                        "content": text,
                        "url": f"file://{pdf_path}",
                        "title": title,
                        "source": self.source_id,
                        "categories": [],
                        "content_hash": content_hash,
                        "char_count": len(text),
                        "pdf_file": pdf_path.name,
                    })

            except Exception as e:
                errors.append(f"{pdf_path.name}: {str(e)}")

        # Index to vector store
        indexed_count = 0
        index_data = None
        if documents:
            try:
                from vectordb.store import VectorStore
                store = VectorStore()
                result = store.add_documents(documents, return_index_data=True)
                indexed_count = result["count"]
                index_data = result["index_data"]
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to index to vector store: {e}",
                    "indexed_count": 0,
                    "errors": errors
                }

        # Save v2 format files
        if indexed_count > 0:
            save_source_metadata(self.output_folder, self.source_id, documents,
                                source_type="pdf")
            save_documents_metadata(self.output_folder, self.source_id, documents)
            if index_data:
                save_embeddings_file(self.output_folder, self.source_id, index_data)
            save_distribution_manifest(self.output_folder, self.source_id, documents,
                                      source_type="pdf",
                                      backup_info={"type": "pdf", "total_pdfs": total_pdfs})

            # Also save legacy format
            self._save_metadata(documents, total_pdfs)

        # Update master metadata
        self._update_master(len(documents))

        return {
            "success": True,
            "indexed_count": indexed_count,
            "total_pdfs": total_pdfs,
            "total_chunks": len(documents),
            "skipped": skipped,
            "errors": errors
        }

    def _save_metadata(self, documents: list, total_pdfs: int) -> None:
        """Save metadata file for this source"""
        from datetime import datetime

        metadata = {
            "source_id": self.source_id,
            "source_type": "pdf",
            "indexed_at": datetime.now().isoformat(),
            "total_documents": len(documents),
            "total_pdfs": total_pdfs,
            "total_chars": sum(d.get("char_count", 0) for d in documents),
            "documents": {d["id"]: {
                "title": d["title"],
                "url": d["url"],
                "content_hash": d["content_hash"],
                "char_count": d["char_count"],
                "pdf_file": d.get("pdf_file", ""),
            } for d in documents}
        }

        metadata_file = self.output_folder / f"{self.source_id}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_file}")

    def _update_master(self, doc_count: int) -> None:
        """Update master metadata file"""
        from datetime import datetime

        try:
            # Find master file in backup path
            master_file = self.output_folder.parent / "_master.json"
            if not master_file.exists():
                master_file = self.output_folder / "_master.json"

            master = {"version": 2, "sources": {}}
            if master_file.exists():
                with open(master_file, 'r', encoding='utf-8') as f:
                    master = json.load(f)

            if "sources" not in master:
                master["sources"] = {}

            master["sources"][self.source_id] = {
                "count": doc_count,
                "type": "pdf",
                "last_sync": datetime.now().isoformat()
            }
            master["last_updated"] = datetime.now().isoformat()

            with open(master_file, 'w', encoding='utf-8') as f:
                json.dump(master, f, indent=2)

        except Exception as e:
            print(f"Warning: Failed to update master metadata: {e}")


def index_pdf_folder(pdf_path: str, source_id: str, limit: int = 1000,
                     progress_callback: Optional[Callable] = None,
                     skip_existing: bool = True,
                     backup_folder: str = None) -> Dict:
    """
    Convenience function to index a PDF file or folder.

    Args:
        pdf_path: Path to PDF file or folder containing PDFs
        source_id: Source identifier for the content
        limit: Maximum PDFs to index
        progress_callback: Progress callback function
        skip_existing: Skip PDFs already in database
        backup_folder: Folder to save output files

    Returns:
        Dict with results
    """
    indexer = PDFIndexer(pdf_path, source_id, backup_folder=backup_folder)
    return indexer.index(limit=limit, progress_callback=progress_callback,
                        skip_existing=skip_existing)


def index_zim_file(zim_path: str, source_id: str, limit: int = 1000,
                   progress_callback: Optional[Callable] = None,
                   backup_folder: str = None) -> Dict:
    """
    Convenience function to index a ZIM file.

    Args:
        zim_path: Path to ZIM file
        source_id: Source identifier for the content
        limit: Maximum articles to index
        progress_callback: Progress callback function
        backup_folder: Folder to save output files (index, metadata, manifest)

    Returns:
        Dict with results
    """
    indexer = ZIMIndexer(zim_path, source_id, backup_folder=backup_folder)
    return indexer.index(limit=limit, progress_callback=progress_callback)


def index_html_backup(backup_path: str, source_id: str, limit: int = 1000,
                      progress_callback: Optional[Callable] = None,
                      skip_existing: bool = True,
                      backup_folder: str = None) -> Dict:
    """
    Convenience function to index an HTML backup.

    Args:
        backup_path: Path to backup folder (containing manifest.json)
        source_id: Source identifier for the content
        limit: Maximum pages to index
        progress_callback: Progress callback function
        skip_existing: Skip pages already in database
        backup_folder: Folder to save output files (index, metadata, manifest)

    Returns:
        Dict with results
    """
    indexer = HTMLBackupIndexer(backup_path, source_id, backup_folder=backup_folder)
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
