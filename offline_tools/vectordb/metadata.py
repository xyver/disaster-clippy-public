"""
Metadata index for fast tracking of ingested documents.
Stores lightweight metadata in JSON for quick comparisons without querying the vector DB.

Structure:
  BACKUP_PATH/
    _master.json                          # Master index summarizing all sources
    {source_id}/
      {source_id}_metadata.json           # Per-source metadata
    ...
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from urllib.parse import urlparse


def extract_domain(url: str) -> str:
    """Extract domain name from URL for use as source identifier"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        # Use just the main domain name (e.g., "appropedia" from "appropedia.org")
        parts = domain.split(".")
        if len(parts) >= 2:
            return parts[-2]  # e.g., "appropedia" from "www.appropedia.org"
        return domain or "unknown"
    except:
        return "unknown"


# Document type constants
DOC_TYPE_GUIDE = "guide"        # Step-by-step instructions, DIY plans
DOC_TYPE_ARTICLE = "article"    # General information, explanations
DOC_TYPE_PRODUCT = "product"    # Commercial products, things to buy
DOC_TYPE_ACADEMIC = "academic"  # Scholarly papers, research, studies

# Source priority constants (for search ranking)
SOURCE_PRIORITY_LOCAL = "LOCAL"           # Partner-specific local data (highest priority)
SOURCE_PRIORITY_GOVERNMENT = "GOVERNMENT" # Government reports (FEMA, EPA, etc.)
SOURCE_PRIORITY_GUIDE = "GUIDE"           # Educational guides (Appropedia, etc.)
SOURCE_PRIORITY_RESEARCH = "RESEARCH"     # Academic papers
SOURCE_PRIORITY_ARTICLE = "ARTICLE"       # General articles
SOURCE_PRIORITY_PRODUCT = "PRODUCT"       # Commercial products


def classify_doc_type(title: str, content: str = "", categories: List[str] = None) -> str:
    """
    Classify a document as guide, article, product, or academic based on title/content patterns.

    - guide: Step-by-step instructions, how-to, build plans, DIY
    - product: Commercial products, manufacturers, vendors, things to buy
    - academic: Scholarly papers, research studies, scientific publications
    - article: General information, overviews, explanations (default)

    Args:
        title: Document title
        content: Document content (optional, first ~500 chars used)
        categories: Document categories (optional)

    Returns:
        One of: "guide", "article", "product", "academic"
    """
    title_lower = title.lower()
    content_sample = content[:500].lower() if content else ""
    cats_lower = [c.lower() for c in (categories or [])]

    # Academic indicators (check early - very specific patterns)
    academic_keywords = [
        "abstract", "methodology", "literature review", "hypothesis",
        "peer-reviewed", "journal", "et al", "findings suggest",
        "statistical analysis", "p-value", "confidence interval",
        "research shows", "study found", "data collected", "participants",
        "experiment", "control group", "bibliography", "references",
        "doi:", "issn", "volume", "issue", "pp.", "cited"
    ]
    academic_title_patterns = [
        "study of", "analysis of", "research on", "investigation of",
        "evaluation of", "assessment of", "review of", "survey of",
        "comparison of", "effects of", "impact of", "role of",
        "towards", "a novel", "an empirical", "experimental"
    ]
    academic_categories = [
        "research", "academic", "papers", "studies", "science",
        "journal", "publications", "peer-reviewed"
    ]

    # Check categories for academic indicators
    if any(ac in cat for cat in cats_lower for ac in academic_categories):
        return DOC_TYPE_ACADEMIC

    # Check title for academic patterns
    if any(p in title_lower for p in academic_title_patterns):
        return DOC_TYPE_ACADEMIC

    # Check content for academic keywords (need several to confirm)
    academic_keyword_count = sum(1 for kw in academic_keywords if kw in content_sample)
    if academic_keyword_count >= 3:
        return DOC_TYPE_ACADEMIC

    # Product indicators
    product_keywords = [
        "manufacturer", "vendor", "company", "buy", "purchase", "price",
        "commercial", "product", "brand", "store", "shop", "order",
        "supplier", "distributor", "retailer", "for sale", "available from"
    ]
    product_title_patterns = [
        "inc.", "llc", "ltd", "corp", "company", "industries"
    ]

    # Check categories for product indicators
    product_categories = ["manufacturers", "vendors", "products", "commercial"]
    if any(pc in cat for cat in cats_lower for pc in product_categories):
        return DOC_TYPE_PRODUCT

    # Check title for company names
    if any(p in title_lower for p in product_title_patterns):
        return DOC_TYPE_PRODUCT

    # Check content for product keywords (higher threshold)
    product_keyword_count = sum(1 for kw in product_keywords if kw in content_sample)
    if product_keyword_count >= 3:
        return DOC_TYPE_PRODUCT

    # Guide indicators
    guide_keywords = [
        "how to", "step by step", "instructions", "build", "construct",
        "make your own", "diy", "plans", "tutorial", "guide", "assembly",
        "materials needed", "tools needed", "step 1", "step 2", "first,",
        "then,", "next,", "finally,", "procedure", "method", "construction"
    ]
    guide_title_patterns = [
        "how to", "build", "make", "construct", "diy", "plans", "design",
        "tutorial", "project", "homemade", "simple", "easy",
        # Device names are usually design/build guides
        "cooker", "oven", "heater", "collector", "dryer", "still",
        "reflector", "parabolic", "panel", "trough", "box cooker",
        "solar cooker", "solar oven", "solar heater", "solar collector",
        "hexayurt", "shelter", "filter", "biogas", "digester"
    ]
    guide_categories = ["plans", "construction", "diy", "how-to", "projects"]

    # Check title for guide patterns
    if any(p in title_lower for p in guide_title_patterns):
        return DOC_TYPE_GUIDE

    # Check categories for guide indicators
    if any(gc in cat for cat in cats_lower for gc in guide_categories):
        return DOC_TYPE_GUIDE

    # Check content for guide keywords
    guide_keyword_count = sum(1 for kw in guide_keywords if kw in content_sample)
    if guide_keyword_count >= 2:
        return DOC_TYPE_GUIDE

    # Default to article
    return DOC_TYPE_ARTICLE


class MetadataIndex:
    """
    Lightweight JSON-based index of all ingested documents.
    Organized by domain/source for better management.

    - Each source gets its own JSON file (e.g., {source_id}_metadata.json)
    - Master file (_master.json) summarizes all sources
    - Much faster than querying ChromaDB for sync comparisons
    """

    def __init__(self, index_dir: str = None):
        # Use BACKUP_PATH as default
        if index_dir is None:
            import os
            index_dir = os.getenv("BACKUP_PATH", "")
            if not index_dir:
                try:
                    from admin.local_config import get_local_config
                    config = get_local_config()
                    index_dir = config.get_backup_folder() or ""
                except ImportError:
                    pass

        # Allow cloud-only mode (no local backup path)
        # In this mode, MetadataIndex returns empty results
        self._cloud_only = not index_dir
        self._source_cache: Dict[str, Dict[str, Any]] = {}

        if self._cloud_only:
            self.index_dir = None
            self.master_file = None
            self._master = self._empty_master()
        else:
            self.index_dir = Path(index_dir)
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.master_file = self.index_dir / "_master.json"
            self._master = self._load_master()

    def _load_master(self) -> Dict[str, Any]:
        """Load master index from disk"""
        if self.master_file.exists():
            try:
                with open(self.master_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self._empty_master()

        return self._empty_master()

    def _empty_master(self) -> Dict[str, Any]:
        """Create empty master index structure"""
        return {
            "version": 2,
            "last_updated": None,
            "total_documents": 0,
            "total_chars": 0,
            "sources": {}  # source_name -> {count, chars, last_sync, file, topics}
        }

    def _empty_source(self, source: str) -> Dict[str, Any]:
        """Create empty source file structure"""
        return {
            "source": source,
            "version": 1,
            "last_updated": None,
            "total_documents": 0,
            "total_chars": 0,
            "documents": {}  # doc_id -> {title, url, categories, content_hash, scraped_at, char_count}
        }

    def _get_source_file(self, source: str) -> Optional[Path]:
        """Get path to source-specific JSON file"""
        if self._cloud_only:
            return None
        # Sanitize source name for filename
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in source.lower())
        return self.index_dir / f"{safe_name}.json"

    def _load_source(self, source: str) -> Dict[str, Any]:
        """Load source-specific data (with caching)"""
        if source in self._source_cache:
            return self._source_cache[source]

        # Cloud-only mode returns empty source
        if self._cloud_only:
            data = self._empty_source(source)
            self._source_cache[source] = data
            return data

        source_file = self._get_source_file(source)
        if source_file and source_file.exists():
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._source_cache[source] = data
                    return data
            except (json.JSONDecodeError, IOError):
                pass

        data = self._empty_source(source)
        self._source_cache[source] = data
        return data

    def _save_source(self, source: str):
        """Save source-specific data to disk"""
        # No-op in cloud-only mode
        if self._cloud_only:
            return

        if source not in self._source_cache:
            return

        data = self._source_cache[source]
        data["last_updated"] = datetime.utcnow().isoformat()
        data["total_documents"] = len(data["documents"])
        data["total_chars"] = sum(d.get("char_count", 0) for d in data["documents"].values())

        source_file = self._get_source_file(source)
        if source_file:
            with open(source_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def _update_master(self):
        """Recalculate and save master index from all source files"""
        # No-op in cloud-only mode
        if self._cloud_only:
            return

        total_docs = 0
        total_chars = 0

        for source, info in self._master["sources"].items():
            source_data = self._load_source(source)
            count = len(source_data["documents"])
            chars = sum(d.get("char_count", 0) for d in source_data["documents"].values())

            info["count"] = count
            info["chars"] = chars
            info["last_sync"] = source_data.get("last_updated")

            # Extract topics from document titles
            topics = set()
            for doc in source_data["documents"].values():
                title_lower = doc.get("title", "").lower()
                # Simple topic extraction
                for keyword in ["water", "solar", "shelter", "food", "garden", "compost",
                               "sanitation", "energy", "filter", "rainwater", "hexayurt"]:
                    if keyword in title_lower:
                        topics.add(keyword)
            info["topics"] = sorted(list(topics))[:5]  # Top 5 topics

            total_docs += count
            total_chars += chars

        self._master["total_documents"] = total_docs
        self._master["total_chars"] = total_chars
        self._master["last_updated"] = datetime.utcnow().isoformat()

        if self.master_file:
            with open(self.master_file, 'w', encoding='utf-8') as f:
                json.dump(self._master, f, indent=2, ensure_ascii=False)

    def add_document(self, doc: Dict[str, Any]):
        """
        Add a document to the index.

        Args:
            doc: Document dict with keys: id/content_hash, title, url, source, categories, content, doc_type
        """
        doc_id = doc.get("id") or doc.get("content_hash") or str(hash(doc.get("url", "")))
        source = doc.get("source", "unknown")

        # Load source data
        source_data = self._load_source(source)

        # Classify document type if not provided
        doc_type = doc.get("doc_type")
        if not doc_type:
            doc_type = classify_doc_type(
                title=doc.get("title", ""),
                content=doc.get("content", ""),
                categories=doc.get("categories", [])
            )

        # Determine source priority from doc_type
        source_priority_map = {
            DOC_TYPE_GUIDE: "GUIDE",
            DOC_TYPE_ARTICLE: "ARTICLE",
            DOC_TYPE_PRODUCT: "PRODUCT",
            DOC_TYPE_ACADEMIC: "RESEARCH"
        }
        source_priority = source_priority_map.get(doc_type, "ARTICLE")

        # Override with source-based priority (e.g., FEMA = GOVERNMENT)
        source_lower = source.lower()
        if any(gov in source_lower for gov in ["fema", "epa", "calfire", "usda", "gov"]):
            source_priority = "GOVERNMENT"

        # Add document to source file
        source_data["documents"][doc_id] = {
            "title": doc.get("title", "Unknown"),
            "url": doc.get("url", ""),
            "categories": doc.get("categories", []),
            "content_hash": doc.get("content_hash", ""),
            "scraped_at": doc.get("scraped_at", datetime.utcnow().isoformat()),
            "char_count": len(doc.get("content", "")),
            "doc_type": doc_type,
            # License tracking (Phase 1)
            "license": doc.get("license", "Unknown"),
            "license_url": doc.get("license_url", ""),
            "license_verified": doc.get("license_verified", False),
            # Source priority for search ranking
            "source_priority": source_priority,
            # Upstream modification date (if available from source)
            "last_modified": doc.get("last_modified", None),
            # Sync tracking
            "vector_db_synced": doc.get("vector_db_synced", False),
            "sync_timestamp": doc.get("sync_timestamp", None)
        }

        # Ensure source is in master
        if source not in self._master["sources"]:
            source_file = self._get_source_file(source)
            self._master["sources"][source] = {
                "count": 0,
                "chars": 0,
                "last_sync": None,
                "file": source_file.name if source_file else None,
                "topics": []
            }

    def add_documents(self, docs: List[Dict[str, Any]]):
        """Add multiple documents and save once"""
        # Group by source for efficient saving
        by_source: Dict[str, List[Dict]] = {}
        for doc in docs:
            source = doc.get("source", "unknown")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(doc)

        # Add all documents
        for doc in docs:
            self.add_document(doc)

        # Save all modified sources
        for source in by_source.keys():
            self._save_source(source)

        # Update master
        self._update_master()

    def get_titles(self, source: Optional[str] = None) -> Set[str]:
        """Get all document titles (fast lookup for deduplication)"""
        titles = set()

        if source:
            source_data = self._load_source(source)
            return {doc["title"] for doc in source_data["documents"].values()}

        # Get from all sources
        for src in self._master["sources"].keys():
            source_data = self._load_source(src)
            titles.update(doc["title"] for doc in source_data["documents"].values())

        return titles

    def get_urls(self, source: Optional[str] = None) -> Set[str]:
        """Get all document URLs"""
        urls = set()

        if source:
            source_data = self._load_source(source)
            return {doc["url"] for doc in source_data["documents"].values()}

        for src in self._master["sources"].keys():
            source_data = self._load_source(src)
            urls.update(doc["url"] for doc in source_data["documents"].values())

        return urls

    def get_ids(self, source: Optional[str] = None) -> Set[str]:
        """Get all document IDs"""
        ids = set()

        if source:
            source_data = self._load_source(source)
            return set(source_data["documents"].keys())

        for src in self._master["sources"].keys():
            source_data = self._load_source(src)
            ids.update(source_data["documents"].keys())

        return ids

    def has_title(self, title: str, source: Optional[str] = None) -> bool:
        """Check if a title exists in the index"""
        return title in self.get_titles(source)

    def has_url(self, url: str, source: Optional[str] = None) -> bool:
        """Check if a URL exists in the index"""
        return url in self.get_urls(source)

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics from master"""
        return {
            "total_documents": self._master["total_documents"],
            "total_chars": self._master.get("total_chars", 0),
            "last_updated": self._master["last_updated"],
            "sources": self._master["sources"]
        }

    def get_source_breakdown(self) -> Dict[str, int]:
        """Get document count by source"""
        return {source: info["count"] for source, info in self._master["sources"].items()}

    def get_documents_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific source"""
        source_data = self._load_source(source)
        return [
            {"id": doc_id, "source": source, **doc}
            for doc_id, doc in source_data["documents"].items()
        ]

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from all sources"""
        all_docs = []
        for source in self._master["sources"].keys():
            all_docs.extend(self.get_documents_by_source(source))
        return all_docs

    def search_titles(self, query: str, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search documents by title (case-insensitive)"""
        query_lower = query.lower()
        results = []

        sources_to_search = [source] if source else self._master["sources"].keys()

        for src in sources_to_search:
            source_data = self._load_source(src)
            for doc_id, doc in source_data["documents"].items():
                if query_lower in doc["title"].lower():
                    results.append({"id": doc_id, "source": src, **doc})

        return results

    def remove_document(self, doc_id: str, source: Optional[str] = None):
        """Remove a document from the index"""
        sources_to_check = [source] if source else list(self._master["sources"].keys())

        for src in sources_to_check:
            source_data = self._load_source(src)
            if doc_id in source_data["documents"]:
                del source_data["documents"][doc_id]
                self._save_source(src)
                self._update_master()
                return True
        return False

    def clear(self, source: Optional[str] = None):
        """Clear all documents from the index (or just one source)"""
        if source:
            # Clear just one source
            source_data = self._empty_source(source)
            self._source_cache[source] = source_data
            self._save_source(source)
            # Remove from master if empty
            if source in self._master["sources"]:
                del self._master["sources"][source]
            self._update_master()
        else:
            # Clear everything
            for src in list(self._master["sources"].keys()):
                source_file = self._get_source_file(src)
                if source_file.exists():
                    source_file.unlink()
            self._master = self._empty_master()
            self._source_cache = {}
            with open(self.master_file, 'w', encoding='utf-8') as f:
                json.dump(self._master, f, indent=2, ensure_ascii=False)

    def sync_from_vectordb(self, vectordb_docs: List[Dict[str, Any]]):
        """
        Rebuild index from vector database contents.
        Useful if index gets out of sync.
        """
        self.clear()
        self.add_documents(vectordb_docs)

    def list_sources(self) -> List[str]:
        """List all source names"""
        return list(self._master["sources"].keys())

    def reclassify_all_documents(self, content_lookup: Optional[Dict[str, str]] = None):
        """
        Reclassify all documents with doc_type based on title/categories.

        Args:
            content_lookup: Optional dict mapping doc_id -> content for better classification
        """
        print("Reclassifying all documents...")
        total_reclassified = 0
        type_counts = {DOC_TYPE_GUIDE: 0, DOC_TYPE_ARTICLE: 0, DOC_TYPE_PRODUCT: 0, DOC_TYPE_ACADEMIC: 0}

        for source in self._master["sources"].keys():
            source_data = self._load_source(source)
            source_count = 0

            for doc_id, doc in source_data["documents"].items():
                content = ""
                if content_lookup and doc_id in content_lookup:
                    content = content_lookup[doc_id]

                doc_type = classify_doc_type(
                    title=doc.get("title", ""),
                    content=content,
                    categories=doc.get("categories", [])
                )
                doc["doc_type"] = doc_type
                type_counts[doc_type] += 1
                source_count += 1

            self._save_source(source)
            total_reclassified += source_count
            print(f"  {source}: {source_count} documents reclassified")

        self._update_master()
        print(f"\nReclassification complete!")
        print(f"  Total: {total_reclassified}")
        print(f"  Guides: {type_counts[DOC_TYPE_GUIDE]}")
        print(f"  Articles: {type_counts[DOC_TYPE_ARTICLE]}")
        print(f"  Products: {type_counts[DOC_TYPE_PRODUCT]}")
        print(f"  Academic: {type_counts[DOC_TYPE_ACADEMIC]}")

        return type_counts

    def get_doc_type_counts(self, source: Optional[str] = None) -> Dict[str, int]:
        """Get count of documents by type"""
        counts = {DOC_TYPE_GUIDE: 0, DOC_TYPE_ARTICLE: 0, DOC_TYPE_PRODUCT: 0, DOC_TYPE_ACADEMIC: 0}

        sources_to_check = [source] if source else self._master["sources"].keys()

        for src in sources_to_check:
            source_data = self._load_source(src)
            for doc in source_data["documents"].values():
                doc_type = doc.get("doc_type", DOC_TYPE_ARTICLE)
                if doc_type in counts:
                    counts[doc_type] += 1

        return counts

    def get_documents_by_type(self, doc_type: str, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all documents of a specific type"""
        results = []
        sources_to_check = [source] if source else self._master["sources"].keys()

        for src in sources_to_check:
            source_data = self._load_source(src)
            for doc_id, doc in source_data["documents"].items():
                if doc.get("doc_type", DOC_TYPE_ARTICLE) == doc_type:
                    results.append({"id": doc_id, "source": src, **doc})

        return results

    def export_summary(self) -> str:
        """Export a human-readable summary"""
        lines = [
            "=" * 60,
            "METADATA INDEX SUMMARY",
            "=" * 60,
            f"Total Documents: {self._master['total_documents']}",
            f"Total Characters: {self._master.get('total_chars', 0):,}",
            f"Last Updated: {self._master['last_updated'] or 'Never'}",
            f"Index Location: {self.index_dir}",
            "",
            "Sources:",
        ]

        for source, info in self._master["sources"].items():
            topics_str = ", ".join(info.get("topics", [])[:3]) or "none detected"
            lines.append(f"  - {source}: {info['count']} docs, {info.get('chars', 0):,} chars")
            lines.append(f"      Topics: {topics_str}")
            lines.append(f"      File: {info.get('file', 'N/A')}")
            lines.append(f"      Last sync: {info.get('last_sync', 'Never')}")

        lines.append("")
        lines.append("Recent Documents (across all sources):")

        # Get recent documents from all sources
        all_docs = []
        for source in self._master["sources"].keys():
            source_data = self._load_source(source)
            for doc_id, doc in source_data["documents"].items():
                all_docs.append((source, doc_id, doc))

        # Sort by scraped_at
        sorted_docs = sorted(
            all_docs,
            key=lambda x: x[2].get("scraped_at", ""),
            reverse=True
        )[:10]

        for source, doc_id, doc in sorted_docs:
            lines.append(f"  - [{source}] {doc['title'][:45]} ({doc.get('char_count', 0):,} chars)")

        return "\n".join(lines)

    def export_source_summary(self, source: str) -> str:
        """Export summary for a specific source"""
        source_data = self._load_source(source)

        lines = [
            "=" * 60,
            f"SOURCE: {source.upper()}",
            "=" * 60,
            f"Total Documents: {len(source_data['documents'])}",
            f"Total Characters: {sum(d.get('char_count', 0) for d in source_data['documents'].values()):,}",
            f"Last Updated: {source_data.get('last_updated') or 'Never'}",
            f"File: {self._get_source_file(source)}",
            "",
            "Documents:",
        ]

        # Sort by title
        sorted_docs = sorted(
            source_data["documents"].items(),
            key=lambda x: x[1].get("title", "")
        )

        for doc_id, doc in sorted_docs:
            cats = ", ".join(doc.get("categories", [])[:2]) or "none"
            lines.append(f"  - {doc['title'][:50]}")
            lines.append(f"      URL: {doc.get('url', 'N/A')}")
            lines.append(f"      Categories: {cats}")

        return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    index = MetadataIndex()
    print(index.export_summary())
