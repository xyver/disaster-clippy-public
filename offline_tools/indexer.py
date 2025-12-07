"""
Indexer module for ingesting content from ZIM files, HTML backups, and PDFs.

Outputs schema files:
    - _manifest.json (source identity + distribution info)
    - _metadata.json (document metadata for quick scanning)
    - _index.json (full document content for display)
    - _vectors.json (embeddings only)
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

from offline_tools.vectordb import VectorStore, MetadataIndex
from .schemas import (
    get_manifest_file, get_metadata_file, get_index_file, get_vectors_file,
    get_backup_manifest_file, CURRENT_SCHEMA_VERSION
)


# =============================================================================
# V3 OUTPUT FUNCTIONS
# =============================================================================

def save_manifest(output_folder: Path, source_id: str, documents: List[Dict],
                  source_type: str = "unknown", base_url: str = "",
                  license_info: str = "Unknown", backup_info: Dict = None,
                  zim_metadata: Dict = None) -> Optional[Path]:
    """
    Save source manifest file (_manifest.json).

    Combines source identity and distribution info into one file.
    Preserves user-edited fields if file exists.

    Args:
        zim_metadata: Optional metadata from ZIM header_fields to auto-populate
                     name, description, creator, license, tags, etc.
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

        manifest_file = output_folder / get_manifest_file()

        # Load existing to preserve user-edited fields
        existing = {}
        if manifest_file.exists():
            try:
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                pass

        # Fields to preserve from existing file (user edits take precedence)
        preserved = ["name", "description", "license", "license_verified",
                     "attribution", "base_url", "tags", "created_at", "version",
                     "language", "publisher"]

        # Calculate file sizes
        metadata_file = output_folder / get_metadata_file()
        index_file = output_folder / get_index_file()
        vectors_file = output_folder / get_vectors_file()

        manifest = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "source_id": source_id,
            "name": source_id.replace('_', ' ').title(),
            "description": "",
            "license": license_info,
            "license_verified": False,
            "attribution": "",
            "base_url": base_url,
            "tags": [],
            "total_docs": len(documents),
            "total_chars": total_chars,
            "categories": categories,
            "version": "1.0.0",
            "source_type": source_type,
            "has_backup": backup_info is not None or (output_folder / "pages").exists(),
            "has_metadata": True,
            "has_index": True,
            "has_vectors": True,
            "backup_info": backup_info or {},
            "metadata_size_bytes": metadata_file.stat().st_size if metadata_file.exists() else 0,
            "index_size_bytes": index_file.stat().st_size if index_file.exists() else 0,
            "vectors_size_bytes": vectors_file.stat().st_size if vectors_file.exists() else 0,
            "total_size_bytes": 0,  # Updated below
            "created_at": existing.get("created_at", datetime.now().isoformat()),
            "last_backup": existing.get("last_backup", ""),
            "last_indexed": datetime.now().isoformat(),
        }

        # Apply ZIM metadata if available (auto-populate from ZIM header_fields)
        if zim_metadata:
            if zim_metadata.get("name"):
                manifest["name"] = zim_metadata["name"]
            if zim_metadata.get("description"):
                manifest["description"] = zim_metadata["description"]
            if zim_metadata.get("license"):
                manifest["license"] = zim_metadata["license"]
            if zim_metadata.get("source_url"):
                manifest["base_url"] = zim_metadata["source_url"]
            if zim_metadata.get("tags"):
                manifest["tags"] = zim_metadata["tags"]
            if zim_metadata.get("creator"):
                manifest["attribution"] = zim_metadata["creator"]
            # Store additional ZIM-specific fields
            if zim_metadata.get("language"):
                manifest["language"] = zim_metadata["language"]
            if zim_metadata.get("publisher"):
                manifest["publisher"] = zim_metadata["publisher"]
            if zim_metadata.get("date"):
                manifest["zim_date"] = zim_metadata["date"]

        # Preserve user-edited fields (override ZIM metadata if user has edited)
        for field in preserved:
            if field in existing and existing[field]:
                manifest[field] = existing[field]

        # Calculate total size
        manifest["total_size_bytes"] = (
            manifest["metadata_size_bytes"] +
            manifest["index_size_bytes"] +
            manifest["vectors_size_bytes"]
        )

        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Saved manifest: {manifest_file}")
        return manifest_file

    except Exception as e:
        print(f"Warning: Failed to save manifest: {e}")
        return None


def save_metadata(output_folder: Path, source_id: str,
                  documents: List[Dict]) -> Optional[Path]:
    """
    Save document metadata file (_metadata.json).

    Contains per-document metadata for quick scanning without loading content.
    """
    try:
        print(f"[save_metadata] Saving metadata for {len(documents)} documents to {output_folder}")

        doc_metadata = {}
        total_chars = 0

        for doc in documents:
            doc_id = doc.get("id", doc.get("content_hash", ""))
            char_count = doc.get("char_count", len(doc.get("content", "")))
            total_chars += char_count

            doc_metadata[doc_id] = {
                "title": doc.get("title", "Unknown"),
                "url": doc.get("url", ""),
                "local_url": doc.get("local_url", ""),  # Local URL for offline use
                "content_hash": doc.get("content_hash", ""),
                "char_count": char_count,
                "categories": doc.get("categories", []),
                "doc_type": doc.get("doc_type", "article"),
                "scraped_at": doc.get("scraped_at", datetime.now().isoformat()),
            }

        metadata = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "source_id": source_id,
            "document_count": len(doc_metadata),
            "total_chars": total_chars,
            "last_updated": datetime.now().isoformat(),
            "documents": doc_metadata
        }

        metadata_file = output_folder / get_metadata_file()
        print(f"[save_metadata] Writing to: {metadata_file}")

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[save_metadata] Saved: {metadata_file} ({len(doc_metadata)} documents)")
        return metadata_file

    except Exception as e:
        print(f"[save_metadata] ERROR: Failed to save metadata: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_index(output_folder: Path, source_id: str,
               documents: List[Dict]) -> Optional[Path]:
    """
    Save full content index file (_index.json).

    Contains full document content for display and scanning.
    """
    try:
        doc_content = {}

        for doc in documents:
            doc_id = doc.get("id", doc.get("content_hash", ""))
            doc_content[doc_id] = {
                "title": doc.get("title", "Unknown"),
                "url": doc.get("url", ""),
                "local_url": doc.get("local_url", ""),  # Local URL for offline use
                "content": doc.get("content", ""),
                "categories": doc.get("categories", []),
                "doc_type": doc.get("doc_type", "article"),
            }

        index_data = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "source_id": source_id,
            "document_count": len(doc_content),
            "created_at": datetime.now().isoformat(),
            "documents": doc_content
        }

        index_file = output_folder / get_index_file()
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False)  # No indent - can be large

        print(f"Saved index: {index_file} ({len(doc_content)} documents)")
        return index_file

    except Exception as e:
        print(f"Warning: Failed to save index: {e}")
        return None


def save_vectors(output_folder: Path, source_id: str,
                 index_data: Dict, embedding_model: str = "text-embedding-3-small") -> Optional[Path]:
    """
    Save vectors file (_vectors.json).

    Contains only embedding vectors, no content duplication.
    """
    try:
        ids = index_data.get("ids", [])
        embeddings = index_data.get("embeddings", [])

        vectors = {}
        dimensions = 0
        for i in range(len(ids)):
            if i < len(embeddings) and embeddings[i]:
                vectors[ids[i]] = embeddings[i]
                if dimensions == 0:
                    dimensions = len(embeddings[i])

        vectors_data = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "source_id": source_id,
            "embedding_model": embedding_model,
            "dimensions": dimensions or 1536,
            "document_count": len(vectors),
            "created_at": datetime.now().isoformat(),
            "vectors": vectors
        }

        vectors_file = output_folder / get_vectors_file()
        with open(vectors_file, 'w', encoding='utf-8') as f:
            json.dump(vectors_data, f)  # No indent - keep compact

        print(f"Saved vectors: {vectors_file} ({len(vectors)} vectors)")
        return vectors_file

    except Exception as e:
        print(f"Warning: Failed to save vectors: {e}")
        return None


def save_all_outputs(output_folder: Path, source_id: str, documents: List[Dict],
                     index_data: Dict = None, source_type: str = "unknown",
                     base_url: str = "", license_info: str = "Unknown",
                     backup_info: Dict = None, zim_metadata: Dict = None) -> Dict[str, Path]:
    """
    Save all output files for a source.

    Args:
        zim_metadata: Optional metadata extracted from ZIM header_fields
                     (name, description, creator, license, tags, etc.)

    Returns dict of saved file paths.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    saved = {}

    # Save metadata first (needed for size calculation in manifest)
    metadata_path = save_metadata(output_folder, source_id, documents)
    if metadata_path:
        saved["metadata"] = metadata_path

    # Save full content index
    index_path = save_index(output_folder, source_id, documents)
    if index_path:
        saved["index"] = index_path

    # Save vectors if we have them
    if index_data:
        vectors_path = save_vectors(output_folder, source_id, index_data)
        if vectors_path:
            saved["vectors"] = vectors_path

    # Save manifest last (references other files for size info)
    manifest_path = save_manifest(output_folder, source_id, documents,
                                  source_type=source_type, base_url=base_url,
                                  license_info=license_info, backup_info=backup_info,
                                  zim_metadata=zim_metadata)
    if manifest_path:
        saved["manifest"] = manifest_path

    # Update master metadata
    try:
        from offline_tools.packager import update_master_metadata
        update_master_metadata(source_id, {
            "document_count": len(documents),
            "total_chars": sum(d.get("char_count", 0) for d in documents),
            "last_updated": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Warning: Failed to update master metadata: {e}")

    return saved


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove scripts, styles, navigation, ads
    for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
        tag.decompose()

    # Try to find main content area
    main_content = (
        soup.find('main') or
        soup.find('article') or
        soup.find('div', {'id': 'content'}) or
        soup.find('div', {'class': 'mw-parser-output'})
    )

    if main_content:
        text = main_content.get_text(separator=' ', strip=True)
    else:
        text = soup.get_text(separator=' ', strip=True)

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return ' '.join(lines)


def get_title_from_html(html_content: str, fallback_url: str = "") -> str:
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
    if fallback_url:
        return fallback_url.split('/')[-1].replace('_', ' ').replace('.html', '')

    return "Untitled"


# =============================================================================
# LANGUAGE DETECTION
# =============================================================================

# Common language codes and their URL patterns
# Extended list to catch more languages in multi-language ZIM files
LANGUAGE_PATTERNS = {
    'en': ['en', 'eng', 'english'],
    'es': ['es', 'esp', 'spanish', 'espanol'],
    'fr': ['fr', 'fra', 'french', 'francais'],
    'de': ['de', 'deu', 'german', 'deutsch'],
    'pt': ['pt', 'por', 'portuguese', 'portugues'],
    'it': ['it', 'ita', 'italian', 'italiano'],
    'ru': ['ru', 'rus', 'russian'],
    'zh': ['zh', 'zho', 'chinese', 'mandarin'],
    'ja': ['ja', 'jpn', 'japanese'],
    'ko': ['ko', 'kor', 'korean'],
    'ar': ['ar', 'ara', 'arabic'],
    'hi': ['hi', 'hin', 'hindi'],
    # Additional languages commonly found in humanitarian/DIY ZIMs
    'vi': ['vi', 'vie', 'vietnamese'],
    'th': ['th', 'tha', 'thai'],
    'id': ['id', 'ind', 'indonesian'],
    'ms': ['ms', 'msa', 'malay'],
    'tl': ['tl', 'fil', 'tagalog', 'filipino'],
    'sw': ['sw', 'swa', 'swahili'],
    'ht': ['ht', 'hat', 'haitian', 'creole', 'kreyol'],
    'bn': ['bn', 'ben', 'bengali', 'bangla'],
    'ne': ['ne', 'nep', 'nepali'],
    'ur': ['ur', 'urd', 'urdu'],
    'fa': ['fa', 'fas', 'persian', 'farsi'],
    'tr': ['tr', 'tur', 'turkish'],
    'pl': ['pl', 'pol', 'polish'],
    'nl': ['nl', 'nld', 'dutch'],
    'uk': ['uk', 'ukr', 'ukrainian'],
    'ro': ['ro', 'ron', 'romanian'],
    'el': ['el', 'ell', 'greek'],
    'he': ['he', 'heb', 'hebrew'],
    'am': ['am', 'amh', 'amharic'],
    'si': ['si', 'sin', 'sinhala', 'sinhalese'],
    'ta': ['ta', 'tam', 'tamil'],
    'te': ['te', 'tel', 'telugu'],
    'my': ['my', 'mya', 'burmese', 'myanmar'],
    'km': ['km', 'khm', 'khmer', 'cambodian'],
    'lo': ['lo', 'lao', 'laotian'],
}

# Set of known base language codes for quick lookup
KNOWN_LANGUAGE_CODES = set(LANGUAGE_PATTERNS.keys())

# Title suffixes that indicate language
# Extended to match common patterns in ZIM files
LANGUAGE_TITLE_SUFFIXES = {
    'en': ['(English)', '(EN)'],
    'es': ['(Spanish)', '(ES)', '(Espanol)'],
    'fr': ['(French)', '(FR)', '(Francais)'],
    'de': ['(German)', '(DE)', '(Deutsch)'],
    'pt': ['(Portuguese)', '(PT)', '(Portugues)'],
    'it': ['(Italian)', '(IT)', '(Italiano)'],
    'ru': ['(Russian)', '(RU)'],
    'zh': ['(Chinese)', '(ZH)', '(Mandarin)'],
    'ja': ['(Japanese)', '(JA)'],
    'ko': ['(Korean)', '(KO)'],
    'ar': ['(Arabic)', '(AR)'],
    'hi': ['(Hindi)', '(HI)'],
    # Additional languages
    'vi': ['(Vietnamese)', '(VI)'],
    'th': ['(Thai)', '(TH)'],
    'id': ['(Indonesian)', '(ID)'],
    'ms': ['(Malay)', '(MS)'],
    'tl': ['(Tagalog)', '(TL)', '(Filipino)'],
    'sw': ['(Swahili)', '(SW)'],
    'ht': ['(Haitian Creole)', '(HT)', '(Haitian)', '(Creole)', '(Kreyol)'],
    'bn': ['(Bengali)', '(BN)', '(Bangla)'],
    'ne': ['(Nepali)', '(NE)'],
    'ur': ['(Urdu)', '(UR)'],
    'fa': ['(Persian)', '(FA)', '(Farsi)'],
    'tr': ['(Turkish)', '(TR)'],
    'pl': ['(Polish)', '(PL)'],
    'nl': ['(Dutch)', '(NL)'],
    'uk': ['(Ukrainian)', '(UK)'],
    'ro': ['(Romanian)', '(RO)'],
    'el': ['(Greek)', '(EL)'],
    'he': ['(Hebrew)', '(HE)'],
    'am': ['(Amharic)', '(AM)'],
    'si': ['(Sinhala)', '(SI)', '(Sinhalese)'],
    'ta': ['(Tamil)', '(TA)'],
    'te': ['(Telugu)', '(TE)'],
    'my': ['(Burmese)', '(MY)', '(Myanmar)'],
    'km': ['(Khmer)', '(KM)', '(Cambodian)'],
    'lo': ['(Lao)', '(LO)', '(Laotian)'],
}


def detect_article_language(url: str, title: str = "") -> Optional[str]:
    """
    Detect language from article URL or title.

    Checks for:
    - URL path segments like /en/, /es/, /fr/
    - Title suffixes like (Spanish), (French)
    - Language keywords anywhere in title (for translations)

    Returns:
        ISO language code (e.g., 'en', 'es', 'fr') or None if not detected
    """
    url_lower = url.lower()
    title_lower = title.lower()

    # Check URL patterns like /en/, /es/, /english/, etc.
    for lang_code, patterns in LANGUAGE_PATTERNS.items():
        for pattern in patterns:
            # Match path segments like /en/ or /english/
            if f'/{pattern}/' in url_lower:
                return lang_code
            # Match at end of domain like .en or _en
            if url_lower.endswith(f'/{pattern}') or f'_{pattern}/' in url_lower:
                return lang_code

    # Smart detection for hyphenated BCP 47 language codes like pt-br, zh-hans, zh-hant
    # Extract base language code from hyphenated patterns and check against known codes
    import re
    hyphenated_match = re.search(r'/([a-z]{2,3})-[a-z]{2,}/', url_lower)
    if hyphenated_match:
        base_code = hyphenated_match.group(1)
        if base_code in KNOWN_LANGUAGE_CODES:
            return base_code

    # Check title suffixes in parentheses
    for lang_code, suffixes in LANGUAGE_TITLE_SUFFIXES.items():
        for suffix in suffixes:
            if suffix.lower() in title_lower:
                return lang_code

    # Also check for language keywords at end of title or after dash/colon
    # e.g., "Solar cooker - Vietnamese", "Water filter: Thai version"
    for lang_code, patterns in LANGUAGE_PATTERNS.items():
        for pattern in patterns:
            # Check for language name at end of title after separator
            if re.search(rf'[-:/]\s*{pattern}\s*$', title_lower):
                return lang_code
            # Check for language name in parentheses anywhere
            if re.search(rf'\({pattern}\)', title_lower):
                return lang_code

    return None


def should_include_article(url: str, title: str, language_filter: Optional[str],
                           debug: bool = False) -> bool:
    """
    Check if article should be included based on language filter.

    Args:
        url: Article URL
        title: Article title
        language_filter: Target language code (e.g., 'en') or None for all
        debug: If True, print debug info for filtered articles

    Returns:
        True if article should be included
    """
    if not language_filter:
        return True  # No filter, include all

    detected = detect_article_language(url, title)

    # If no language detected, include by default (might be main content)
    if detected is None:
        return True

    # Check if detected language matches filter
    matches = detected == language_filter.lower()

    if debug and not matches:
        print(f"[lang_filter] EXCLUDED: '{title[:50]}' detected={detected}, filter={language_filter}")

    return matches


# =============================================================================
# ZIM INDEXER
# =============================================================================

class ZIMIndexer:
    """Indexes content from ZIM files into the vector database."""

    def __init__(self, zim_path: str, source_id: str, backup_folder: str = None):
        self.zim_path = Path(zim_path)
        self.source_id = source_id
        self.output_folder = Path(backup_folder) if backup_folder else self.zim_path.parent
        self.output_folder.mkdir(parents=True, exist_ok=True)

        if not self.zim_path.exists():
            raise FileNotFoundError(f"ZIM file not found: {zim_path}")

    def _extract_zim_metadata(self, header_fields: Dict) -> Dict:
        """
        Extract metadata from ZIM header_fields.

        Common ZIM metadata fields:
        - Title: Human-readable name
        - Description: Content description
        - Creator: Organization that created the content
        - Publisher: Organization that created the ZIM
        - Date: Creation date (YYYY-MM-DD)
        - Language: ISO language code (e.g., 'eng', 'fra')
        - License: License information
        - Tags: Semicolon-separated tags
        - Source: Original URL of the content

        Returns dict with normalized metadata for manifest.
        """
        metadata = {
            "name": header_fields.get("Title", "") or header_fields.get("Name", ""),
            "description": header_fields.get("Description", ""),
            "creator": header_fields.get("Creator", ""),
            "publisher": header_fields.get("Publisher", ""),
            "date": header_fields.get("Date", ""),
            "language": header_fields.get("Language", ""),
            "license": header_fields.get("License", ""),
            "source_url": header_fields.get("Source", ""),
            "tags": [],
        }

        # Parse tags if present (semicolon-separated in ZIM format)
        tags_str = header_fields.get("Tags", "")
        if tags_str:
            metadata["tags"] = [t.strip() for t in tags_str.split(";") if t.strip()]

        # Clean up empty strings
        for key in list(metadata.keys()):
            if metadata[key] == "":
                metadata[key] = None

        print(f"ZIM metadata extracted: title='{metadata.get('name')}', "
              f"license='{metadata.get('license')}', language='{metadata.get('language')}'")

        return metadata

    def _build_online_url(self, zim_path: str, zim_metadata: Optional[Dict] = None) -> str:
        """
        Build online URL from ZIM path.

        ZIM files often store paths with full domains (e.g., www.ready.gov/be-informed).
        This method detects such paths and converts them to proper https:// URLs.

        Fallback order:
        1. If path looks like a full URL (www.*, contains domain), prepend https://
        2. If zim_metadata has source_url, use base_url + article name
        3. Return None (caller should use local_url)
        """
        import re

        # Check if ZIM path contains a full URL (common in ZIM files)
        # Match patterns like: www.example.com/..., subdomain.domain.tld/...
        url_pattern = r'^(www\.|[a-z0-9-]+\.(gov|com|org|net|edu|io|co|info|wiki)[/\.])'
        if re.match(url_pattern, zim_path, re.IGNORECASE):
            # Path is a full URL, just add protocol
            return f"https://{zim_path}"

        # Try base_url from ZIM metadata
        base_url = zim_metadata.get("source_url", "") if zim_metadata else ""
        if base_url:
            if not base_url.endswith('/'):
                base_url += '/'
            # For wiki-style URLs, use last path segment as article name
            article_name = zim_path.split('/')[-1] if '/' in zim_path else zim_path
            return f"{base_url}{article_name}"

        # No online URL available
        return None

    def index(self, limit: int = 1000, progress_callback: Optional[Callable] = None,
              language_filter: Optional[str] = None,
              clear_existing: bool = False) -> Dict:
        """
        Index content from ZIM file into vector database.

        If _metadata.json exists (from Generate Metadata step), uses that document
        list instead of re-scanning the entire ZIM. This is faster and ensures
        consistency with any language filtering applied during metadata generation.

        Args:
            limit: Maximum number of articles to index
            progress_callback: Function(current, total, message) for progress updates
            language_filter: ISO language code to filter by (e.g., 'en', 'es').
                           Only used if _metadata.json doesn't exist.
                           If metadata exists, its filtering is already applied.
            clear_existing: If True, delete all existing documents for this source
                          from ChromaDB before indexing (for force reindex).

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
        deleted_count = 0

        # Clear existing documents from ChromaDB if requested (force reindex)
        if clear_existing:
            print(f"[ZIMIndexer] Force reindex requested - clearing existing documents for '{self.source_id}'")
            try:
                store = VectorStore()
                deleted_count = store.delete_source(self.source_id)
                if deleted_count > 0:
                    print(f"[ZIMIndexer] Cleared {deleted_count} existing documents for {self.source_id}")
                else:
                    print(f"[ZIMIndexer] No existing documents found to clear for {self.source_id}")
            except Exception as e:
                print(f"[ZIMIndexer] Warning: Could not clear existing documents: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[ZIMIndexer] Skip existing mode - not clearing documents")

        # Check for existing _metadata.json - use it if available
        from .schemas import get_metadata_file
        metadata_path = self.output_folder / get_metadata_file()
        use_metadata = metadata_path.exists()
        metadata_docs = {}

        if use_metadata:
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata_data = json.load(f)
                metadata_docs = metadata_data.get("documents", {})
                if metadata_docs:
                    print(f"[ZIMIndexer] Using existing _metadata.json with {len(metadata_docs)} documents")
                    print(f"[ZIMIndexer] Language filtering was applied during metadata generation")
                else:
                    use_metadata = False
                    print(f"[ZIMIndexer] _metadata.json exists but is empty, falling back to full scan")
            except Exception as e:
                use_metadata = False
                print(f"[ZIMIndexer] Could not read _metadata.json: {e}, falling back to full scan")

        try:
            # Progress is split: 0-50% extraction, 50-100% embeddings
            def report_extraction_progress(current, total, message):
                if progress_callback:
                    # Scale to 0-50%
                    scaled_progress = int((current / max(total, 1)) * 50)
                    progress_callback(scaled_progress, 100, message)

            def report_embedding_progress(current, total, message):
                if progress_callback:
                    # Scale to 50-100%
                    scaled_progress = 50 + int((current / max(total, 1)) * 50)
                    progress_callback(scaled_progress, 100, message)

            report_extraction_progress(0, 100, "Opening ZIM file...")

            zim = ZIMFile(str(self.zim_path), 'utf-8')
            article_count = zim.header_fields.get('articleCount', 0)
            print(f"ZIM file contains {article_count} articles total")

            # Extract ZIM metadata from header_fields
            zim_metadata = self._extract_zim_metadata(zim.header_fields)

            indexed = 0
            skipped = 0
            language_filtered = 0
            seen_ids = set()

            if use_metadata:
                # === FAST PATH: Use existing metadata ===
                # Only fetch content for documents already in _metadata.json
                docs_to_process = list(metadata_docs.items())[:limit]
                target_count = len(docs_to_process)
                report_extraction_progress(0, target_count, f"Processing {target_count} documents from metadata...")

                for idx, (doc_id, doc_info) in enumerate(docs_to_process):
                    try:
                        zim_index = doc_info.get("zim_index")
                        if zim_index is None:
                            # No zim_index stored - skip this document
                            skipped += 1
                            continue

                        article = zim.get_article_by_id(zim_index)
                        if article is None:
                            skipped += 1
                            continue

                        content = article.data
                        if isinstance(content, bytes):
                            content = content.decode('utf-8', errors='ignore')

                        if not content or len(content) < 100:
                            skipped += 1
                            continue

                        # Use metadata for title/url, extract fresh content
                        title = doc_info.get("title", "")
                        zim_url = doc_info.get("zim_url", doc_info.get("url", ""))

                        text = extract_text_from_html(content)
                        if len(text) < 50:
                            skipped += 1
                            continue

                        # Build URLs - local for offline viewer, online for Pinecone
                        local_url = f"/zim/{self.source_id}/{zim_url}"
                        online_url = self._build_online_url(zim_url, zim_metadata) or local_url

                        # Use consistent doc_id
                        final_doc_id = hashlib.md5(f"{self.source_id}:{zim_url}".encode()).hexdigest()

                        if final_doc_id in seen_ids:
                            skipped += 1
                            continue
                        seen_ids.add(final_doc_id)

                        documents.append({
                            "id": final_doc_id,
                            "content": text[:50000],
                            "title": title,
                            "url": online_url,
                            "local_url": local_url,
                            "source": self.source_id,
                            "categories": [],
                            "content_hash": hashlib.md5(text.encode()).hexdigest(),
                            "scraped_at": datetime.now().isoformat(),
                            "char_count": len(text),
                            "doc_type": "article",
                        })

                        indexed += 1

                        if indexed % 10 == 0:
                            report_extraction_progress(indexed, target_count,
                                            f"Extracting: {title[:50]}...")

                    except Exception as e:
                        errors.append(f"Error processing article {zim_index}: {str(e)}")
                        continue

                print(f"[ZIMIndexer] Extracted {len(documents)} articles using metadata (skipped {skipped})")

            else:
                # === LEGACY PATH: Full ZIM scan (fallback if no metadata) ===
                target_count = min(limit, article_count)
                report_extraction_progress(0, target_count, f"Found {article_count} articles in ZIM")

                if language_filter:
                    print(f"Language filter: {language_filter} (only indexing {language_filter} articles)")

                for i in range(article_count):
                    if indexed >= limit:
                        break

                    try:
                        article = zim.get_article_by_id(i)
                        if article is None:
                            skipped += 1
                            continue

                        # Only want HTML articles
                        mimetype = getattr(article, 'mimetype', '')
                        if 'text/html' not in str(mimetype).lower():
                            skipped += 1
                            continue

                        content = article.data
                        if isinstance(content, bytes):
                            content = content.decode('utf-8', errors='ignore')

                        if not content or len(content) < 100:
                            skipped += 1
                            continue

                        url = getattr(article, 'url', '') or f"article_{i}"
                        title = getattr(article, 'title', '') or get_title_from_html(content, url)

                        # Filter by language if specified
                        # Enable debug for first 10 filtered articles to see what's being excluded
                        if language_filter and not should_include_article(
                            url, title, language_filter, debug=(language_filtered < 10)
                        ):
                            language_filtered += 1
                            continue

                        # Skip special pages
                        if any(x in url.lower() for x in ['special:', 'file:', 'category:', 'template:', 'mediawiki:', '-/', 'favicon']):
                            skipped += 1
                            continue

                        text = extract_text_from_html(content)
                        if len(text) < 50:
                            skipped += 1
                            continue

                        # Build URLs - local for offline viewer, online for Pinecone
                        local_url = f"/zim/{self.source_id}/{url}"
                        online_url = self._build_online_url(url, zim_metadata) or local_url

                        doc_id = hashlib.md5(f"{self.source_id}:{url}".encode()).hexdigest()

                        if doc_id in seen_ids:
                            skipped += 1
                            continue
                        seen_ids.add(doc_id)

                        documents.append({
                            "id": doc_id,
                            "content": text[:50000],
                            "title": title,
                            "url": online_url,  # Online URL for Pinecone
                            "local_url": local_url,  # Local URL for ChromaDB
                            "source": self.source_id,
                            "categories": [],
                            "content_hash": hashlib.md5(text.encode()).hexdigest(),
                            "scraped_at": datetime.now().isoformat(),
                            "char_count": len(text),
                            "doc_type": "article",
                        })

                        indexed += 1

                        if indexed % 10 == 0:
                            report_extraction_progress(indexed, target_count,
                                            f"Extracting: {title[:50]}...")

                    except Exception as e:
                        errors.append(f"Error processing article {i}: {str(e)}")
                        continue

                lang_info = f", {language_filtered} filtered by language" if language_filtered > 0 else ""
                print(f"Extracted {len(documents)} articles from ZIM (skipped {skipped}{lang_info})")

            zim.close()

            if not documents:
                return {
                    "success": False,
                    "error": "No articles found in ZIM file",
                    "indexed_count": 0,
                    "errors": errors
                }

            # Add to vector store - embeddings phase (50-100%)
            report_embedding_progress(0, len(documents), f"Computing embeddings for {len(documents)} documents...")

            print("Adding documents to vector store...")
            store = VectorStore()
            result = store.add_documents(documents, return_index_data=True,
                                        progress_callback=report_embedding_progress)
            count = result["count"]
            index_data = result["index_data"]

            # Save all output files with ZIM metadata
            if count > 0:
                # Build backup_info with ZIM metadata
                backup_info = {
                    "type": "zim",
                    "path": str(self.zim_path),
                    "size_mb": round(self.zim_path.stat().st_size / (1024*1024), 2),
                    "zim_metadata": zim_metadata,  # Full extracted metadata
                }

                save_all_outputs(
                    self.output_folder, self.source_id, documents, index_data,
                    source_type="zim",
                    base_url=zim_metadata.get("source_url") or "",
                    license_info=zim_metadata.get("license") or "Unknown",
                    backup_info=backup_info,
                    zim_metadata=zim_metadata,  # Pass for manifest population
                )

            return {
                "success": True,
                "indexed_count": count,
                "total_extracted": len(documents),
                "skipped": skipped,
                "deleted_existing": deleted_count,
                "language_filtered": language_filtered,
                "language_filter": language_filter,
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


# =============================================================================
# HTML BACKUP INDEXER
# =============================================================================

class HTMLBackupIndexer:
    """Indexes content from HTML backup folders into the vector database."""

    def __init__(self, backup_path: str, source_id: str, backup_folder: str = None):
        self.backup_path = Path(backup_path)
        self.source_id = source_id
        self.pages_dir = self.backup_path / "pages"
        self.output_folder = Path(backup_folder) if backup_folder else self.backup_path
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Find manifest file
        self.manifest_path = None
        candidates = [
            self.backup_path / get_backup_manifest_file(),  # New: backup_manifest.json
            self.backup_path / f"{source_id}_backup_manifest.json",  # Legacy v2
            self.backup_path / f"{source_id}_manifest.json",  # Legacy v1
            self.backup_path / "manifest.json",  # Very old
        ]
        for candidate in candidates:
            if candidate.exists():
                self.manifest_path = candidate
                break

        if not self.backup_path.exists():
            raise FileNotFoundError(f"Backup folder not found: {backup_path}")

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
        pages = {}

        # Load manifest or scan pages folder
        if self.manifest_path and self.manifest_path.exists():
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            pages = manifest.get("pages", {})

        # Fallback: scan pages folder directly
        if not pages and self.pages_dir.exists():
            print(f"No pages in manifest, scanning {self.pages_dir}...")
            for html_file in self.pages_dir.glob("*.html"):
                filename = html_file.name
                url_path = filename.replace(".html", "").replace("_", "/")
                pages[f"/{url_path}"] = {
                    "filename": filename,
                    "title": filename.replace(".html", "").replace("_", " ")
                }
            print(f"Found {len(pages)} HTML files")

        if not pages:
            return {
                "success": False,
                "error": "No pages found in manifest or pages folder",
                "indexed_count": 0
            }

        print(f"Found {len(pages)} pages to index")

        # Get existing IDs if skipping
        existing_ids = set()
        if skip_existing:
            store = VectorStore()
            existing_ids = store.get_existing_ids()
            print(f"Found {len(existing_ids)} already indexed documents")

        if progress_callback:
            progress_callback(0, min(limit, len(pages)), "Loading pages...")

        indexed = 0
        skipped = 0

        for url, page_info in pages.items():
            if indexed >= limit:
                break

            doc_id = hashlib.md5(f"{self.source_id}:{url}".encode()).hexdigest()

            if skip_existing and doc_id in existing_ids:
                skipped += 1
                continue

            try:
                filename = page_info.get("filename")
                title = page_info.get("title", "Untitled")

                if not filename:
                    errors.append(f"No filename for {url}")
                    continue

                html_path = self.pages_dir / filename
                if not html_path.exists():
                    errors.append(f"File not found: {filename}")
                    continue

                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                text = extract_text_from_html(html_content)
                if len(text) < 50:
                    skipped += 1
                    continue

                # Build URLs - local_url for offline serving, url for online
                local_url = f"/backup/{self.source_id}/{filename}"

                # Online URL: use base_url + relative path
                # base_url loaded later, so we store the relative URL in 'url'
                # and construct online URL when base_url is known
                online_url = url  # Will be updated below with base_url

                documents.append({
                    "id": doc_id,
                    "content": text[:50000],
                    "title": title,
                    "url": online_url,  # Online URL (relative, combined with base_url below)
                    "local_url": local_url,  # Local URL for offline serving
                    "source": self.source_id,
                    "categories": [],
                    "content_hash": hashlib.md5(text.encode()).hexdigest(),
                    "scraped_at": datetime.now().isoformat(),
                    "char_count": len(text),
                    "doc_type": "article",
                })

                indexed += 1

                if progress_callback and indexed % 10 == 0:
                    progress_callback(indexed, min(limit, len(pages) - skipped),
                                    f"Processing: {title[:50]}...")

            except Exception as e:
                errors.append(f"Error processing {url}: {str(e)}")

        print(f"Prepared {len(documents)} documents (skipped {skipped})")

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
            progress_callback(len(documents), len(documents), "Computing embeddings...")

        print("Adding documents to vector store...")
        store = VectorStore()
        result = store.add_documents(documents, return_index_data=True)
        count = result["count"]
        index_data = result["index_data"]

        # Get base_url from manifest
        base_url = ""
        try:
            if self.manifest_path and self.manifest_path.exists():
                with open(self.manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                base_url = manifest.get("base_url", "")
        except Exception:
            pass

        # Update document URLs with base_url for online use
        if base_url:
            for doc in documents:
                relative_url = doc.get("url", "")
                # Construct full online URL from base_url + relative path
                if relative_url and not relative_url.startswith(('http://', 'https://')):
                    if base_url.endswith('/') and relative_url.startswith('/'):
                        doc["url"] = base_url + relative_url[1:]
                    elif not base_url.endswith('/') and not relative_url.startswith('/'):
                        doc["url"] = base_url + '/' + relative_url
                    else:
                        doc["url"] = base_url + relative_url

        # Save all output files (v3 format)
        if count > 0:
            save_all_outputs(
                self.output_folder, self.source_id, documents, index_data,
                source_type="html", base_url=base_url
            )

        return {
            "success": True,
            "indexed_count": count,
            "total_processed": len(documents),
            "skipped": skipped,
            "errors": errors,
            "output_folder": str(self.output_folder)
        }


# =============================================================================
# PDF INDEXER
# =============================================================================

class PDFIndexer:
    """Indexes content from PDF files into the vector database."""

    def __init__(self, source_path: str, source_id: str, backup_folder: str = None):
        self.source_path = Path(source_path)
        self.source_id = source_id

        if backup_folder:
            self.output_folder = Path(backup_folder)
        elif self.source_path.is_file():
            self.output_folder = self.source_path.parent
        else:
            self.output_folder = self.source_path
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Check for PDF library
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
                "No PDF library available. Install: pip install pymupdf"
            )

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract text from a PDF file"""
        if self.has_pymupdf:
            import fitz
            doc = fitz.open(str(pdf_path))
            text_parts = [page.get_text() for page in doc]
            doc.close()
            return "\n".join(text_parts)
        elif self.has_pypdf:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            text_parts = [page.extract_text() or "" for page in reader.pages]
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
                }
            elif self.has_pypdf:
                from pypdf import PdfReader
                reader = PdfReader(str(pdf_path))
                if reader.metadata:
                    return {
                        "title": reader.metadata.get("/Title", ""),
                        "author": reader.metadata.get("/Author", ""),
                    }
        except Exception:
            pass
        return {}

    def _get_pdf_files(self) -> list:
        """Get list of PDF files to process"""
        if self.source_path.is_file() and self.source_path.suffix.lower() == '.pdf':
            return [self.source_path]
        elif self.source_path.is_dir():
            return list(self.source_path.glob("*.pdf"))
        return []

    def index(self, limit: int = 1000, progress_callback: Optional[Callable] = None,
              skip_existing: bool = True, chunk_size: int = 4000) -> Dict:
        """
        Index PDF content into vector database.

        Args:
            limit: Maximum number of PDFs to index
            progress_callback: Function(current, total, message)
            skip_existing: Skip PDFs already in database
            chunk_size: Characters per chunk for long PDFs

        Returns:
            Dict with results
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

        existing_ids = set()
        if skip_existing:
            try:
                store = VectorStore()
                existing_ids = store.get_existing_ids()
            except:
                pass

        for i, pdf_path in enumerate(pdf_files):
            if progress_callback:
                progress_callback(i + 1, total_pdfs, f"Processing {pdf_path.name}")

            try:
                text = self._extract_text(pdf_path)
                if len(text) < 100:
                    errors.append(f"{pdf_path.name}: Too little text")
                    continue

                metadata = self._extract_metadata(pdf_path)
                title = metadata.get("title") or pdf_path.stem.replace('_', ' ').replace('-', ' ').title()

                content_hash = hashlib.md5(text.encode()).hexdigest()[:12]
                doc_id = f"{self.source_id}_{content_hash}"

                if doc_id in existing_ids:
                    skipped += 1
                    continue

                # Build URLs - online points to API endpoint (served from R2), local to file
                pdf_filename = pdf_path.name
                online_url = f"/api/pdf/{self.source_id}/{pdf_filename}"
                local_url = f"file://{pdf_path}"

                # Chunk long PDFs
                if len(text) > chunk_size:
                    chunks = [text[j:j+chunk_size] for j in range(0, len(text), chunk_size)]
                    for idx, chunk in enumerate(chunks):
                        chunk_id = f"{doc_id}_chunk{idx}"
                        if chunk_id in existing_ids:
                            continue
                        documents.append({
                            "id": chunk_id,
                            "content": chunk,
                            "url": f"{online_url}#chunk-{idx}",  # Online URL for Pinecone
                            "local_url": f"{local_url}#chunk-{idx}",  # Local file path
                            "title": f"{title} (Part {idx + 1}/{len(chunks)})",
                            "source": self.source_id,
                            "categories": [],
                            "content_hash": f"{content_hash}_{idx}",
                            "char_count": len(chunk),
                            "doc_type": "research",
                        })
                else:
                    documents.append({
                        "id": doc_id,
                        "content": text,
                        "url": online_url,  # Online URL for Pinecone
                        "local_url": local_url,  # Local file path
                        "title": title,
                        "source": self.source_id,
                        "categories": [],
                        "content_hash": content_hash,
                        "char_count": len(text),
                        "doc_type": "research",
                    })

            except Exception as e:
                errors.append(f"{pdf_path.name}: {str(e)}")

        # Index to vector store
        indexed_count = 0
        index_data = None
        if documents:
            try:
                store = VectorStore()
                result = store.add_documents(documents, return_index_data=True)
                indexed_count = result["count"]
                index_data = result["index_data"]
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to index: {e}",
                    "indexed_count": 0,
                    "errors": errors
                }

        # Save v3 format files
        if indexed_count > 0:
            save_all_outputs(
                self.output_folder, self.source_id, documents, index_data,
                source_type="pdf",
                backup_info={"type": "pdf", "total_pdfs": total_pdfs}
            )

        return {
            "success": True,
            "indexed_count": indexed_count,
            "total_pdfs": total_pdfs,
            "total_chunks": len(documents),
            "skipped": skipped,
            "errors": errors
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def index_zim_file(zim_path: str, source_id: str, limit: int = 1000,
                   progress_callback: Optional[Callable] = None,
                   backup_folder: str = None,
                   language_filter: Optional[str] = None,
                   clear_existing: bool = False) -> Dict:
    """
    Index a ZIM file.

    Args:
        zim_path: Path to the ZIM file
        source_id: Source identifier
        limit: Maximum articles to index
        progress_callback: Progress function(current, total, message)
        backup_folder: Output folder for index files
        language_filter: ISO language code to filter (e.g., 'en', 'es').
                        Only articles in this language will be indexed.
        clear_existing: If True, delete existing documents for this source
                       from ChromaDB before indexing (for force reindex).
    """
    indexer = ZIMIndexer(zim_path, source_id, backup_folder=backup_folder)
    return indexer.index(limit=limit, progress_callback=progress_callback,
                        language_filter=language_filter,
                        clear_existing=clear_existing)


def index_html_backup(backup_path: str, source_id: str, limit: int = 1000,
                      progress_callback: Optional[Callable] = None,
                      skip_existing: bool = True,
                      backup_folder: str = None) -> Dict:
    """Index an HTML backup folder."""
    indexer = HTMLBackupIndexer(backup_path, source_id, backup_folder=backup_folder)
    return indexer.index(limit=limit, progress_callback=progress_callback,
                        skip_existing=skip_existing)


def index_pdf_folder(pdf_path: str, source_id: str, limit: int = 1000,
                     progress_callback: Optional[Callable] = None,
                     skip_existing: bool = True,
                     backup_folder: str = None) -> Dict:
    """Index PDF files."""
    indexer = PDFIndexer(pdf_path, source_id, backup_folder=backup_folder)
    return indexer.index(limit=limit, progress_callback=progress_callback,
                        skip_existing=skip_existing)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python indexer.py <zim|html|pdf> <path> [source_id] [limit]")
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
    elif index_type == "pdf":
        result = index_pdf_folder(path, source_id, limit, progress)
    else:
        print(f"Unknown type: {index_type}")
        sys.exit(1)

    print("\nResult:")
    print(json.dumps(result, indent=2))
