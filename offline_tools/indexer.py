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
    get_backup_manifest_file, CURRENT_SCHEMA_VERSION,
    html_filename_to_url, html_filename_to_title
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
        # CRITICAL: created_from and zim_path must be preserved for ZIM-imported sources
        preserved = ["name", "description", "license", "license_verified",
                     "attribution", "base_url", "tags", "created_at", "version",
                     "language", "publisher", "created_from", "zim_path"]

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

            doc_entry = {
                "title": doc.get("title", "Unknown"),
                "url": doc.get("url", ""),
                "local_url": doc.get("local_url", ""),  # Local URL for offline use
                "content_hash": doc.get("content_hash", ""),
                "char_count": char_count,
                "categories": doc.get("categories", []),
                "doc_type": doc.get("doc_type", "article"),
                "scraped_at": doc.get("scraped_at", datetime.now().isoformat()),
            }
            # Include internal links if present
            if doc.get("internal_links"):
                doc_entry["internal_links"] = doc["internal_links"]
            doc_metadata[doc_id] = doc_entry

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
            doc_entry = {
                "title": doc.get("title", "Unknown"),
                "url": doc.get("url", ""),
                "local_url": doc.get("local_url", ""),  # Local URL for offline use
                "content": doc.get("content", ""),
                "categories": doc.get("categories", []),
                "doc_type": doc.get("doc_type", "article"),
            }
            # Include internal links if present
            if doc.get("internal_links"):
                doc_entry["internal_links"] = doc["internal_links"]
            doc_content[doc_id] = doc_entry

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
                 index_data: Dict, embedding_model: str = "text-embedding-3-small",
                 dimension: int = None) -> Optional[Path]:
    """
    Save vectors file (_vectors.json or _vectors_768.json).

    Contains only embedding vectors, no content duplication.

    Args:
        output_folder: Where to save the file
        source_id: Source identifier
        index_data: Dict with 'ids' and 'embeddings' lists
        embedding_model: Name of the embedding model used
        dimension: If specified, uses dimension-specific filename (768 or 1536)

    Returns:
        Path to saved file, or None on error
    """
    try:
        ids = index_data.get("ids", [])
        embeddings = index_data.get("embeddings", [])

        vectors = {}
        detected_dim = 0
        for i in range(len(ids)):
            if i < len(embeddings) and embeddings[i]:
                vectors[ids[i]] = embeddings[i]
                if detected_dim == 0:
                    detected_dim = len(embeddings[i])

        # Use detected dimension if not specified
        if dimension is None:
            dimension = detected_dim or 1536

        vectors_data = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "source_id": source_id,
            "embedding_model": embedding_model,
            "dimensions": dimension,
            "document_count": len(vectors),
            "created_at": datetime.now().isoformat(),
            "vectors": vectors
        }

        # Use dimension-specific filename
        vectors_file = output_folder / get_vectors_file(dimension)
        with open(vectors_file, 'w', encoding='utf-8') as f:
            json.dump(vectors_data, f)  # No indent - keep compact

        print(f"Saved vectors: {vectors_file} ({len(vectors)} vectors, {dimension}-dim)")
        return vectors_file

    except Exception as e:
        print(f"Warning: Failed to save vectors: {e}")
        return None


def generate_768_vectors(output_folder: Path, source_id: str,
                         documents: List[Dict],
                         progress_callback: Callable = None) -> Optional[Path]:
    """
    Generate 768-dim embeddings for offline use and save to _vectors_768.json.

    This creates the offline-compatible vectors using the local embedding model
    (all-mpnet-base-v2). Use this after initial indexing with 1536-dim to create
    the pack-downloadable version.

    Args:
        output_folder: Source folder to save to
        source_id: Source identifier
        documents: List of document dicts with 'id' and 'content' keys
        progress_callback: Optional function(current, total, message)

    Returns:
        Path to saved _vectors_768.json, or None on error
    """
    from offline_tools.embeddings import EmbeddingService

    try:
        # Force local model for 768-dim embeddings
        embedding_service = EmbeddingService(model="all-mpnet-base-v2")

        if not embedding_service.is_available():
            print(f"Error: Local embedding model not available")
            return None

        # Check dimension
        dim = embedding_service.get_dimension()
        if dim != 768:
            print(f"Warning: Expected 768-dim, got {dim}-dim")

        print(f"Generating 768-dim embeddings for {len(documents)} documents...")

        # Extract content
        ids = []
        contents = []
        for doc in documents:
            doc_id = doc.get("id", doc.get("content_hash", ""))
            content = doc.get("content", "")
            if doc_id and content:
                ids.append(doc_id)
                contents.append(content)

        if not contents:
            print("No content to embed")
            return None

        # Generate embeddings
        embeddings = embedding_service.embed_batch(
            contents,
            progress_callback=progress_callback
        )

        if not embeddings:
            print("Failed to generate embeddings")
            return None

        # Save as 768-dim vectors file
        index_data = {
            "ids": ids,
            "embeddings": embeddings
        }

        return save_vectors(
            output_folder, source_id, index_data,
            embedding_model="all-mpnet-base-v2",
            dimension=768
        )

    except Exception as e:
        print(f"Error generating 768-dim vectors: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_all_outputs(output_folder: Path, source_id: str, documents: List[Dict],
                     index_data: Dict = None, source_type: str = "unknown",
                     base_url: str = "", license_info: str = "Unknown",
                     backup_info: Dict = None, zim_metadata: Dict = None,
                     skip_metadata_save: bool = False) -> Dict[str, Path]:
    """
    Save all output files for a source.

    Args:
        zim_metadata: Optional metadata extracted from ZIM header_fields
                     (name, description, creator, license, tags, etc.)
        skip_metadata_save: If True, don't overwrite _metadata.json (preserves
                           metadata from separate "Generate Metadata" step)

    Returns dict of saved file paths.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    saved = {}

    # Save metadata first (needed for size calculation in manifest)
    # Skip if metadata was loaded from existing file (to preserve full document list)
    if not skip_metadata_save:
        metadata_path = save_metadata(output_folder, source_id, documents)
        if metadata_path:
            saved["metadata"] = metadata_path
    else:
        print(f"[save_all_outputs] Skipping metadata save (preserving existing _metadata.json)")

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

def extract_text_lenient(html_content: str) -> str:
    """
    Extract text from HTML using lenient BeautifulSoup parsing.

    This is the UNIFIED extraction method used by both metadata generation
    and indexing to ensure consistent document counts.

    Approach:
    - Removes junk tags (script, style, nav, header, footer, aside, iframe)
    - Does NOT require main/article container (lenient)
    - Extracts all remaining text content
    - Cleans up whitespace

    This replaces both:
    - The regex-based extraction in metadata generation
    - The strict main-content extraction in indexing
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove junk tags that don't contain useful content
    for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
        tag.decompose()

    # Get all remaining text (lenient - don't require main/article container)
    text = soup.get_text(separator=' ', strip=True)

    # Clean up whitespace - normalize spaces and remove empty lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return ' '.join(lines)


def extract_text_from_html(html_content: str) -> str:
    """
    Extract clean text from HTML content.

    DEPRECATED: This function now calls extract_text_lenient() for consistency.
    Keeping the function name for backwards compatibility with existing code.

    Previously tried to find main content areas (main, article, etc.) which
    caused metadata/index count mismatches. Now uses lenient extraction.
    """
    return extract_text_lenient(html_content)


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
    'uk': ['(Ukrainian)'],  # Note: (UK) means United Kingdom, not Ukrainian
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
# LINK EXTRACTION
# =============================================================================

def extract_internal_links_from_html(html_content: str, base_path: str = "") -> List[str]:
    """
    Extract internal links from HTML content.

    Args:
        html_content: Raw HTML string
        base_path: Base path for resolving relative URLs

    Returns:
        List of internal link paths (e.g., ["/wiki/Solar_panel", "/wiki/Energy"])
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find main content area if possible
        content = (
            soup.find("div", class_="mw-parser-output") or
            soup.find("div", id="mw-content-text") or
            soup.find("div", id="content") or
            soup.find("article") or
            soup.body or
            soup
        )

        internal_links = []
        seen = set()

        for link in content.find_all("a", href=True):
            href = link["href"]

            # Skip empty, anchor-only, external, or special links
            if not href or href.startswith("#"):
                continue
            if href.startswith("http://") or href.startswith("https://"):
                continue
            if href.startswith("javascript:") or href.startswith("mailto:") or href.startswith("tel:"):
                continue

            # Skip special wiki pages
            if any(x in href for x in ["/Special:", "/File:", "/Category:",
                                        "/Template:", "/Help:", "/Talk:",
                                        "/User:", "/Wikipedia:", "/Portal:"]):
                continue

            # Clean the path
            path = href.split("#")[0]  # Remove anchor
            path = path.split("?")[0]  # Remove query string

            if not path:
                continue

            # Normalize path
            if not path.startswith("/"):
                # Relative path - could resolve with base_path but keep simple for now
                path = "/" + path

            # Deduplicate
            if path not in seen:
                seen.add(path)
                internal_links.append(path)

        return internal_links

    except Exception as e:
        print(f"Error extracting links: {e}")
        return []


# NOTE: ZIMIndexer class was removed (Dec 2024)
# All sources now use HTMLBackupIndexer - ZIM files must be extracted via ZIM import job first
# The old class spanned ~678 lines (lines 791-1463) and is no longer needed.

# === ZIMIndexer class deleted (Dec 2024) - 678 lines removed ===
# All ZIM sources must be extracted via ZIM import job before indexing
# Now using HTMLBackupIndexer for all indexing operations

# =============================================================================
# HTML BACKUP INDEXER
# =============================================================================

class HTMLBackupIndexer:
    """Indexes content from HTML backup folders into the vector database."""

    def __init__(self, backup_path: str, source_id: str, backup_folder: str = None, dimension: int = 1536):
        self.backup_path = Path(backup_path)
        self.source_id = source_id
        self.pages_dir = self.backup_path / "pages"
        self.output_folder = Path(backup_folder) if backup_folder else self.backup_path
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension

        # Find manifest file
        manifest_path = self.backup_path / get_backup_manifest_file()
        self.manifest_path = manifest_path if manifest_path.exists() else None

        if not self.backup_path.exists():
            raise FileNotFoundError(f"Backup folder not found: {backup_path}")

    def index(self, limit: int = 1000, progress_callback: Optional[Callable] = None,
              skip_existing: bool = True, clear_existing: bool = False) -> Dict:
        """
        Index content from HTML backup into vector database.

        Args:
            limit: Maximum number of pages to index
            progress_callback: Function(current, total, message) for progress updates
            skip_existing: Skip pages already in the database
            clear_existing: Clear existing documents for this source before indexing (force reindex)

        Returns:
            Dict with success status, count, errors
        """
        errors = []
        documents = []
        pages = {}
        is_zim_source = False

        # Load backup manifest for pages list
        if self.manifest_path and self.manifest_path.exists():
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            pages = manifest.get("pages", {})

        # Check _manifest.json for source type (created_from is in main manifest, not backup manifest)
        # ZIM sources (Wikipedia/MediaWiki) use underscores in article URLs
        main_manifest_path = self.backup_path / "_manifest.json"
        if not main_manifest_path.exists():
            main_manifest_path = self.backup_path / "manifest.json"  # legacy fallback
        if main_manifest_path.exists():
            try:
                with open(main_manifest_path, 'r', encoding='utf-8') as f:
                    main_manifest = json.load(f)
                is_zim_source = main_manifest.get("created_from") == "zim_import"
            except Exception:
                pass

        # Fallback: scan pages folder directly
        if not pages and self.pages_dir.exists():
            print(f"No pages in manifest, scanning {self.pages_dir}...")
            for html_file in self.pages_dir.glob("*.html"):
                filename = html_file.name
                # Use centralized filename conversion (preserve underscores for ZIM sources)
                url_path = html_filename_to_url(filename, is_zim_source=is_zim_source)
                pages[url_path] = {
                    "filename": filename,
                    "title": html_filename_to_title(filename)
                }
            print(f"Found {len(pages)} HTML files")

        if not pages:
            return {
                "success": False,
                "error": "No pages found in manifest or pages folder",
                "indexed_count": 0
            }

        print(f"Found {len(pages)} pages to index")

        # Unified progress reporting: map phases to single 0-100% scale
        # 0-5%: Delete existing (if force reindex)
        # 5-60%: Processing HTML pages
        # 60-100%: Adding to vector store (embeddings)
        def report_delete_progress(current, total, message):
            if progress_callback and total > 0:
                # Map to 0-5% range
                unified_progress = int((current / total) * 5)
                progress_callback(unified_progress, 100, message)

        def report_processing_progress(current, total, message):
            if progress_callback and total > 0:
                # Map to 5-60% range (55% of scale)
                unified_progress = 5 + int((current / total) * 55)
                progress_callback(unified_progress, 100, message)

        def report_embedding_progress(current, total, message):
            if progress_callback and total > 0:
                # Map to 60-100% range (40% of scale)
                unified_progress = 60 + int((current / total) * 40)
                progress_callback(unified_progress, 100, message)

        # Clear existing documents if requested (force reindex)
        if clear_existing or not skip_existing:
            print(f"[HTMLBackupIndexer] Force reindex - clearing existing documents for '{self.source_id}' ({self.dimension}-dim)")
            report_delete_progress(0, 1, "Clearing existing documents...")
            try:
                # Use read_only=True to skip loading embedding model - not needed for delete
                store = VectorStore(dimension=self.dimension, read_only=True)
                result = store.delete_by_source(self.source_id)
                deleted_count = result.get("deleted_count", 0) if isinstance(result, dict) else result
                print(f"[HTMLBackupIndexer] Cleared {deleted_count} existing documents")
            except Exception as e:
                print(f"[HTMLBackupIndexer] Warning: Could not clear existing documents: {e}")
            report_delete_progress(1, 1, "Cleared existing documents")

        # Get existing IDs if skipping (only if not force reindex)
        existing_ids = set()
        if skip_existing and not clear_existing:
            store = VectorStore(dimension=self.dimension, read_only=True)
            existing_ids = store.get_existing_ids()
            print(f"Found {len(existing_ids)} already indexed documents ({self.dimension}-dim)")

        report_processing_progress(0, 1, "Loading pages...")

        indexed = 0
        skipped = 0

        for url, page_info in pages.items():
            if limit is not None and indexed >= limit:
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

                if indexed % 10 == 0:
                    total_to_process = len(pages) - skipped
                    if limit is not None:
                        total_to_process = min(limit, total_to_process)
                    report_processing_progress(indexed, total_to_process,
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

        # Get base_url from _manifest.json BEFORE adding to ChromaDB
        # _manifest.json contains source identity info including base_url
        base_url = ""
        try:
            source_manifest_path = self.output_folder / get_manifest_file()
            if source_manifest_path.exists():
                with open(source_manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                base_url = manifest.get("base_url", "")
                if base_url:
                    print(f"[HTMLBackupIndexer] Using base_url: {base_url}")
        except Exception as e:
            print(f"[HTMLBackupIndexer] Could not read base_url from manifest: {e}")

        # Update document URLs with base_url BEFORE adding to ChromaDB
        # This ensures ChromaDB stores full URLs for online use
        if base_url:
            for doc in documents:
                relative_url = doc.get("url", "")
                # Construct full online URL from base_url + relative path
                if relative_url and not relative_url.startswith(('http://', 'https://')):
                    # Check for WARC-style URLs from ZIM files with multiple domains
                    # These can be /www.fema.gov/path OR www.fema.gov/path (no leading slash)
                    # First path component looks like a domain if it contains a dot
                    if relative_url.startswith('/'):
                        first_segment = relative_url[1:].split('/')[0] if '/' in relative_url[1:] else relative_url[1:]
                        # If first segment looks like a domain (contains dot, not starting with dot)
                        if '.' in first_segment and not first_segment.startswith('.'):
                            # WARC-style: convert /www.fema.gov/path to https://www.fema.gov/path
                            path_after_domain = relative_url[1 + len(first_segment):]
                            doc["url"] = f"https://{first_segment}{path_after_domain}"
                            continue
                    else:
                        # Check for domain-like URL without leading slash (www.fema.gov/path)
                        first_segment = relative_url.split('/')[0] if '/' in relative_url else relative_url
                        # If first segment looks like a domain (contains dot, has common TLD pattern)
                        if '.' in first_segment and not first_segment.startswith('.'):
                            # Check for common domain patterns (www., .gov, .org, .com, .edu, .net)
                            if first_segment.startswith('www.') or any(first_segment.endswith(tld) for tld in ['.gov', '.org', '.com', '.edu', '.net', '.io']):
                                # WARC-style: convert www.fema.gov/path to https://www.fema.gov/path
                                doc["url"] = f"https://{relative_url}"
                                continue
                    # Normal relative URL - prepend base_url
                    if base_url.endswith('/') and relative_url.startswith('/'):
                        doc["url"] = base_url + relative_url[1:]
                    elif not base_url.endswith('/') and not relative_url.startswith('/'):
                        doc["url"] = base_url + '/' + relative_url
                    else:
                        doc["url"] = base_url + relative_url

        # Add to vector store (incremental mode)
        report_embedding_progress(0, 1, "Starting embedding generation...")

        print(f"Adding documents to vector store (incremental mode, {self.dimension}-dim)...")
        store = VectorStore(dimension=self.dimension)
        result = store.add_documents_incremental(
            documents,
            source_id=self.source_id,
            batch_size=100,
            return_index_data=True,
            progress_callback=report_embedding_progress
        )
        count = result["count"]
        skipped_existing = result.get("skipped", 0)
        resumed = result.get("resumed", False)
        index_data = result.get("index_data")

        if resumed:
            print(f"[HTMLBackupIndexer] Resumed: {skipped_existing} docs already indexed, {count} new")

        # Save all output files (v3 format)
        # Skip metadata save if _metadata.json already exists (preserves metadata from separate Generate Metadata step)
        if count > 0:
            existing_metadata = (self.output_folder / get_metadata_file()).exists()
            if existing_metadata:
                print(f"[HTMLBackupIndexer] Preserving existing _metadata.json (use Generate Metadata to update)")
            save_all_outputs(
                self.output_folder, self.source_id, documents, index_data,
                source_type="html", base_url=base_url,
                skip_metadata_save=existing_metadata
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

    def __init__(self, source_path: str, source_id: str, backup_folder: str = None, dimension: int = 1536):
        self.source_path = Path(source_path)
        self.source_id = source_id
        self.dimension = dimension

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
        """Extract text from a PDF file (legacy - merges all pages)"""
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

    def _extract_text_with_pages(self, pdf_path: Path) -> List[Dict]:
        """
        Extract text from PDF preserving page boundaries.

        Returns:
            List of dicts: [{"page_num": 1, "text": "...", "char_count": N}, ...]
        """
        pages = []
        if self.has_pymupdf:
            import fitz
            doc = fitz.open(str(pdf_path))
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                pages.append({
                    "page_num": page_num,
                    "text": text,
                    "char_count": len(text)
                })
            doc.close()
        elif self.has_pypdf:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages.append({
                    "page_num": page_num,
                    "text": text,
                    "char_count": len(text)
                })
        return pages

    def _extract_metadata(self, pdf_path: Path) -> dict:
        """Extract metadata from a PDF file (basic - title and author only)"""
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

    def _extract_enhanced_metadata(self, pdf_path: Path) -> dict:
        """
        Extract comprehensive metadata from PDF including page count.

        Returns:
            Dict with title, author, subject, keywords, creator,
            creation_date, page_count, file_size
        """
        result = {
            "title": "",
            "author": "",
            "subject": "",
            "keywords": "",
            "creator": "",
            "creation_date": "",
            "page_count": 0,
            "file_size": 0,
        }
        try:
            result["file_size"] = pdf_path.stat().st_size
        except Exception:
            pass

        try:
            if self.has_pymupdf:
                import fitz
                doc = fitz.open(str(pdf_path))
                metadata = doc.metadata or {}
                result.update({
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "keywords": metadata.get("keywords", ""),
                    "creator": metadata.get("creator", ""),
                    "creation_date": metadata.get("creationDate", ""),
                    "page_count": doc.page_count,
                })
                doc.close()
            elif self.has_pypdf:
                from pypdf import PdfReader
                reader = PdfReader(str(pdf_path))
                result["page_count"] = len(reader.pages)
                if reader.metadata:
                    result.update({
                        "title": reader.metadata.get("/Title", ""),
                        "author": reader.metadata.get("/Author", ""),
                        "subject": reader.metadata.get("/Subject", ""),
                        "keywords": reader.metadata.get("/Keywords", ""),
                        "creator": reader.metadata.get("/Creator", ""),
                        "creation_date": reader.metadata.get("/CreationDate", ""),
                    })
        except Exception:
            pass
        return result

    def _chunk_pages_with_overlap(self, pages: List[Dict], chunk_size: int = 4000,
                                   overlap: int = 300) -> List[Dict]:
        """
        Create chunks from pages with overlap and page tracking.

        Args:
            pages: List from _extract_text_with_pages()
            chunk_size: Target characters per chunk
            overlap: Characters to overlap between chunks

        Returns:
            List of chunk dicts with page_start, page_end, text
        """
        if not pages:
            return []

        chunks = []
        current_text = ""
        current_page_start = pages[0]["page_num"]
        current_page_end = pages[0]["page_num"]

        for page in pages:
            page_text = page["text"]
            page_num = page["page_num"]

            # If adding this page would exceed chunk_size
            if len(current_text) + len(page_text) > chunk_size and current_text:
                # Save current chunk
                chunks.append({
                    "text": current_text,
                    "page_start": current_page_start,
                    "page_end": current_page_end,
                })
                # Start new chunk with overlap from end of previous
                overlap_text = current_text[-overlap:] if overlap and len(current_text) > overlap else ""
                current_text = overlap_text
                current_page_start = page_num

            current_text += page_text + "\n"
            current_page_end = page_num

        # Don't forget the last chunk
        if current_text.strip():
            chunks.append({
                "text": current_text,
                "page_start": current_page_start,
                "page_end": current_page_end,
            })

        return chunks

    def _get_pdf_files(self) -> list:
        """Get list of PDF files to process"""
        if self.source_path.is_file() and self.source_path.suffix.lower() == '.pdf':
            return [self.source_path]
        elif self.source_path.is_dir():
            return list(self.source_path.glob("*.pdf"))
        return []

    def index(self, limit: int = 1000, progress_callback: Optional[Callable] = None,
              skip_existing: bool = True, chunk_size: int = 4000,
              clear_existing: bool = False) -> Dict:
        """
        Index PDF content into vector database.

        Args:
            limit: Maximum number of PDFs to index
            progress_callback: Function(current, total, message)
            skip_existing: Skip PDFs already in database
            chunk_size: Characters per chunk for long PDFs
            clear_existing: Clear existing documents for this source before indexing (force reindex)

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

        # Clear existing documents if requested (force reindex)
        if clear_existing or not skip_existing:
            print(f"[PDFIndexer] Force reindex - clearing existing documents for '{self.source_id}' ({self.dimension}-dim)")
            try:
                # Use read_only=True to skip loading embedding model - not needed for delete
                store = VectorStore(dimension=self.dimension, read_only=True)
                result = store.delete_by_source(self.source_id)
                deleted_count = result.get("deleted_count", 0) if isinstance(result, dict) else result
                print(f"[PDFIndexer] Cleared {deleted_count} existing documents")
            except Exception as e:
                print(f"[PDFIndexer] Warning: Could not clear existing documents: {e}")

        existing_ids = set()
        if skip_existing and not clear_existing:
            try:
                store = VectorStore(dimension=self.dimension, read_only=True)
                existing_ids = store.get_existing_ids()
            except:
                pass

        for i, pdf_path in enumerate(pdf_files):
            if progress_callback:
                progress_callback(i + 1, total_pdfs, f"Processing {pdf_path.name}")

            try:
                # Extract with page awareness
                pages = self._extract_text_with_pages(pdf_path)
                total_text_len = sum(p["char_count"] for p in pages)
                if total_text_len < 100:
                    errors.append(f"{pdf_path.name}: Too little text")
                    continue

                # Get enhanced metadata
                enhanced_meta = self._extract_enhanced_metadata(pdf_path)
                title = enhanced_meta.get("title") or pdf_path.stem.replace('_', ' ').replace('-', ' ').title()
                total_pages = enhanced_meta.get("page_count", len(pages))

                # Full text for hash
                full_text = "\n".join(p["text"] for p in pages)
                content_hash = hashlib.md5(full_text.encode()).hexdigest()[:12]
                doc_id = f"{self.source_id}_{content_hash}"

                if doc_id in existing_ids:
                    skipped += 1
                    continue

                # Build URLs - use /pdf/ endpoint (browser-navigable with #page=N)
                pdf_filename = pdf_path.name
                online_url = f"/pdf/{self.source_id}/{pdf_filename}"
                local_url = f"file://{pdf_path}"

                # Create chunks with page tracking and overlap
                chunks = self._chunk_pages_with_overlap(pages, chunk_size=chunk_size, overlap=300)

                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk{idx}" if len(chunks) > 1 else doc_id
                    if chunk_id in existing_ids:
                        continue

                    # Use page_start for URL (browser navigation)
                    page_url = f"{online_url}#page={chunk['page_start']}"
                    local_page_url = f"{local_url}#page={chunk['page_start']}"

                    # Build title with page range
                    if len(chunks) == 1:
                        chunk_title = title
                    elif chunk["page_start"] == chunk["page_end"]:
                        chunk_title = f"{title} (p. {chunk['page_start']})"
                    else:
                        chunk_title = f"{title} (pp. {chunk['page_start']}-{chunk['page_end']})"

                    documents.append({
                        "id": chunk_id,
                        "content": chunk["text"],
                        "url": page_url,
                        "local_url": local_page_url,
                        "title": chunk_title,
                        "source": self.source_id,
                        "categories": [],
                        "content_hash": f"{content_hash}_{idx}" if len(chunks) > 1 else content_hash,
                        "char_count": len(chunk["text"]),
                        "doc_type": "research",
                        # PDF-specific metadata
                        "parent_pdf": pdf_filename,
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "total_pages": total_pages,
                    })

            except Exception as e:
                errors.append(f"{pdf_path.name}: {str(e)}")

        # Index to vector store (incremental mode)
        indexed_count = 0
        skipped_existing = 0
        index_data = None
        if documents:
            try:
                print(f"[PDFIndexer] Using {self.dimension}-dim embeddings")
                store = VectorStore(dimension=self.dimension)
                result = store.add_documents_incremental(
                    documents,
                    source_id=self.source_id,
                    batch_size=100,
                    return_index_data=True
                )
                indexed_count = result["count"]
                skipped_existing = result.get("skipped", 0)
                index_data = result.get("index_data")

                if result.get("resumed", False):
                    print(f"[PDFIndexer] Resumed: {skipped_existing} chunks already indexed")
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
                   clear_existing: bool = False,
                   resume: bool = False,
                   cancel_checker: Optional[Callable] = None,
                   dimension: int = 1536) -> Dict:
    """
    DEPRECATED: Direct ZIM indexing is no longer supported.

    ZIM files must first be extracted via ZIM import job, then indexed
    using index_html_backup() on the resulting pages/ folder.

    This function now returns an error directing users to the correct workflow.
    """
    return {
        "success": False,
        "error": "Direct ZIM indexing is no longer supported. "
                 "Run ZIM import job first to extract HTML pages, "
                 "then use index_html_backup() on the extracted source.",
        "indexed_count": 0
    }


def index_html_backup(backup_path: str, source_id: str, limit: int = 1000,
                      progress_callback: Optional[Callable] = None,
                      skip_existing: bool = True,
                      backup_folder: str = None,
                      dimension: int = 1536) -> Dict:
    """Index an HTML backup folder."""
    indexer = HTMLBackupIndexer(backup_path, source_id, backup_folder=backup_folder, dimension=dimension)
    # clear_existing=True when skip_existing=False (force reindex)
    return indexer.index(limit=limit, progress_callback=progress_callback,
                        skip_existing=skip_existing,
                        clear_existing=not skip_existing)


def index_pdf_folder(pdf_path: str, source_id: str, limit: int = 1000,
                     progress_callback: Optional[Callable] = None,
                     skip_existing: bool = True,
                     backup_folder: str = None,
                     dimension: int = 1536) -> Dict:
    """Index PDF files."""
    indexer = PDFIndexer(pdf_path, source_id, backup_folder=backup_folder, dimension=dimension)
    # clear_existing=True when skip_existing=False (force reindex)
    return indexer.index(limit=limit, progress_callback=progress_callback,
                        skip_existing=skip_existing,
                        clear_existing=not skip_existing)


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
