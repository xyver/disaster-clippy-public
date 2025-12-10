"""
PDF Collection Management

Manages collections of PDFs with two-level metadata structure:
- Collection level: name, license, topics, permission status
- Document level: title, author, DOI, access level, chunks

Folder Structure:
    pdf_inbox/              <- Incoming unsorted PDFs (triage area)
        document.pdf        <- Individual files
        folder_of_pdfs/     <- Folders get auto-named as potential collections
        archive.zip         <- ZIPs extracted, contents reviewed

    pdf_collections/        <- Organized, processed collections
        flu_preparedness/
            _collection.json
            FluSCIM.pdf
        medical_guides/
            _collection.json
            FirstAid.pdf

Usage:
    manager = PDFCollectionManager()

    # Import from inbox to a collection
    manager.import_from_inbox("document.pdf", collection="flu_preparedness")

    # Or import entire inbox folder as new collection
    manager.import_inbox_folder("folder_of_pdfs", collection="medical_guides")

    # List what's in inbox (needs sorting)
    inbox_items = manager.list_inbox()

    # List organized collections
    collections = manager.list_collections()

    # Process and index a collection
    docs = manager.process_collection("flu_preparedness")
"""

import os
import re
import json
import shutil
import hashlib
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class CollectionMetadata:
    """Metadata for a PDF collection"""
    collection_id: str
    name: str
    description: str = ""
    license: str = "Unknown"
    license_url: str = ""
    license_notes: str = ""
    permission_status: str = "none"  # none, pending, granted, denied
    permission_notes: str = ""
    topics: List[str] = None
    author: str = ""
    publisher: str = ""
    source_url: str = ""  # Original source if downloaded
    created: str = ""
    updated: str = ""

    def __post_init__(self):
        if self.topics is None:
            self.topics = []
        if not self.created:
            self.created = datetime.now().isoformat()
        if not self.updated:
            self.updated = self.created


@dataclass
class PDFDocumentMetadata:
    """Metadata for a single PDF document"""
    filename: str
    title: str
    content_hash: str
    char_count: int
    chunk_count: int = 0
    authors: List[str] = None
    doi: str = ""
    original_url: str = ""
    r2_url: str = ""
    publication_date: str = ""
    access_level: str = "unknown"  # public_domain, open_access, restricted, unknown
    license_override: str = ""  # If different from collection
    added_at: str = ""

    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if not self.added_at:
            self.added_at = datetime.now().isoformat()


# License categories
LICENSE_CATEGORIES = {
    "public_domain": {
        "can_host": True,
        "can_redistribute": True,
        "examples": ["Public Domain", "CC0", "US Government Work"]
    },
    "open_access": {
        "can_host": True,
        "can_redistribute": True,
        "attribution_required": True,
        "examples": ["CC-BY", "CC-BY-SA", "CC-BY-NC", "Open Access"]
    },
    "restricted": {
        "can_host": False,
        "can_redistribute": False,
        "examples": ["All Rights Reserved", "Copyrighted", "Paywalled"]
    },
    "unknown": {
        "can_host": False,
        "can_redistribute": False,
        "examples": ["Unknown", ""]
    }
}


class PDFCollectionManager:
    """
    Manages PDF collections with structured metadata.

    Two-folder structure:
    - inbox_path: Incoming unsorted PDFs (triage area)
    - collections_path: Organized, processed collections with manifests
    """

    def __init__(self, inbox_path: str = None, collections_path: str = None):
        """
        Args:
            inbox_path: Directory for incoming unsorted PDFs
                        Defaults to BACKUP_PATH/pdf_inbox
            collections_path: Directory for organized collections
                              Defaults to BACKUP_PATH/pdf_collections

        Raises:
            ValueError: If backup path is not configured in Settings or .env
        """
        # Determine base backup path - try local_config first, then env
        backup_path = ""
        try:
            from admin.local_config import get_local_config
            config = get_local_config()
            backup_path = config.get_backup_folder() or ""
        except ImportError:
            pass

        if not backup_path:
            backup_path = os.getenv("BACKUP_PATH", "")

        if not backup_path:
            raise ValueError(
                "Backup path not configured. "
                "Please set it in Settings page or BACKUP_PATH in .env"
            )

        backup_base = Path(backup_path)

        # Set inbox path
        if inbox_path:
            self.inbox_path = Path(inbox_path)
        else:
            self.inbox_path = backup_base / "pdf_inbox"

        # Set collections path
        if collections_path:
            self.collections_path = Path(collections_path)
        else:
            self.collections_path = backup_base / "pdf_collections"

        # Ensure paths exist
        self.inbox_path.mkdir(parents=True, exist_ok=True)
        self.collections_path.mkdir(parents=True, exist_ok=True)

        # Legacy alias for backward compatibility
        self.base_path = self.collections_path

    def _get_collection_path(self, collection_id: str) -> Path:
        """Get path to a collection folder"""
        return self.collections_path / collection_id

    def _get_collection_manifest_path(self, collection_id: str) -> Path:
        """Get path to collection's _collection.json"""
        return self._get_collection_path(collection_id) / "_collection.json"

    def _load_collection_metadata(self, collection_id: str) -> Optional[CollectionMetadata]:
        """Load collection metadata from _collection.json"""
        manifest_path = self._get_collection_manifest_path(collection_id)
        if not manifest_path.exists():
            return None

        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return CollectionMetadata(**data.get('collection', data))

    def _save_collection_metadata(self, metadata: CollectionMetadata):
        """Save collection metadata to _collection.json"""
        collection_path = self._get_collection_path(metadata.collection_id)
        collection_path.mkdir(parents=True, exist_ok=True)

        manifest_path = self._get_collection_manifest_path(metadata.collection_id)

        # Load existing documents if any
        documents = {}
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                documents = existing.get('documents', {})

        # Update timestamp
        metadata.updated = datetime.now().isoformat()

        # Save full manifest
        manifest = {
            'collection': asdict(metadata),
            'documents': documents
        }

        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def create_collection(self, collection_id: str,
                          name: str = None,
                          description: str = "",
                          license: str = "Unknown",
                          topics: List[str] = None,
                          **kwargs) -> CollectionMetadata:
        """
        Create a new PDF collection.

        Args:
            collection_id: Unique identifier (folder name)
            name: Display name
            description: Collection description
            license: License type
            topics: List of topic tags
            **kwargs: Additional metadata fields

        Returns:
            CollectionMetadata object
        """
        # Check if already exists
        if self._get_collection_manifest_path(collection_id).exists():
            print(f"Collection '{collection_id}' already exists")
            return self._load_collection_metadata(collection_id)

        metadata = CollectionMetadata(
            collection_id=collection_id,
            name=name or collection_id.replace("_", " ").replace("-", " ").title(),
            description=description,
            license=license,
            topics=topics or [],
            **kwargs
        )

        self._save_collection_metadata(metadata)
        print(f"Created collection: {collection_id}")

        return metadata

    def list_inbox(self) -> Dict[str, Any]:
        """
        List contents of the inbox (unsorted PDFs).

        Returns:
            Dict with 'files', 'folders', 'zips' lists
        """
        result = {
            'files': [],      # Individual PDFs
            'folders': [],    # Folders containing PDFs
            'zips': []        # ZIP archives
        }

        if not self.inbox_path.exists():
            return result

        for item in self.inbox_path.iterdir():
            if item.is_file():
                if item.suffix.lower() == '.pdf':
                    result['files'].append({
                        'name': item.name,
                        'path': str(item),
                        'size': item.stat().st_size,
                        'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })
                elif item.suffix.lower() == '.zip':
                    # Count PDFs inside ZIP
                    pdf_count = 0
                    try:
                        with zipfile.ZipFile(item, 'r') as zf:
                            pdf_count = sum(1 for n in zf.namelist() if n.lower().endswith('.pdf'))
                    except:
                        pass
                    result['zips'].append({
                        'name': item.name,
                        'path': str(item),
                        'size': item.stat().st_size,
                        'pdf_count': pdf_count
                    })
            elif item.is_dir():
                # Count PDFs in folder
                pdf_files = list(item.glob("**/*.pdf"))
                if pdf_files:  # Only list folders with PDFs
                    result['folders'].append({
                        'name': item.name,
                        'path': str(item),
                        'pdf_count': len(pdf_files)
                    })

        return result

    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections with summary info.

        Returns:
            List of collection summaries
        """
        collections = []

        if not self.collections_path.exists():
            return collections

        for item in self.collections_path.iterdir():
            if item.is_dir():
                manifest_path = item / "_collection.json"

                # Count PDFs
                pdf_count = len(list(item.glob("*.pdf")))

                if manifest_path.exists():
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    coll = data.get('collection', {})
                    collections.append({
                        'collection_id': item.name,
                        'name': coll.get('name', item.name),
                        'license': coll.get('license', 'Unknown'),
                        'pdf_count': pdf_count,
                        'document_count': len(data.get('documents', {})),
                        'topics': coll.get('topics', [])
                    })
                else:
                    # Folder without manifest (shouldn't happen in collections)
                    collections.append({
                        'collection_id': item.name,
                        'name': item.name,
                        'license': 'Unknown',
                        'pdf_count': pdf_count,
                        'document_count': 0,
                        'topics': []
                    })

        return collections

    def import_from_inbox(self, item_name: str, collection: str,
                          create_collection: bool = True) -> List[str]:
        """
        Import a file, folder, or ZIP from inbox into a collection.

        Handles:
        - Single PDF file -> moves to collection
        - Folder of PDFs -> moves all PDFs to collection
        - ZIP file -> extracts PDFs to collection, deletes ZIP

        Args:
            item_name: Name of file/folder/zip in inbox
            collection: Collection ID to import into
            create_collection: If True, create collection if it doesn't exist

        Returns:
            List of imported file paths
        """
        item_path = self.inbox_path / item_name

        if not item_path.exists():
            print(f"Not found in inbox: {item_name}")
            return []

        # Ensure collection exists
        if create_collection:
            collection_path = self._get_collection_path(collection)
            if not collection_path.exists():
                self.create_collection(collection)

        imported = []

        if item_path.is_file():
            if item_path.suffix.lower() == '.pdf':
                # Single PDF - move it
                result = self.add_pdf(str(item_path), collection, move=True)
                if result:
                    imported.append(result)

            elif item_path.suffix.lower() == '.zip':
                # ZIP - extract then delete
                imported = self.add_zip(str(item_path), collection)
                if imported:
                    item_path.unlink()  # Delete ZIP after successful extraction
                    print(f"Deleted ZIP after extraction: {item_name}")

        elif item_path.is_dir():
            # Folder - move all PDFs, then delete folder if empty
            imported = self.add_folder(str(item_path), collection, recursive=True, move=True)

            # Check if folder is now empty and remove it
            remaining = list(item_path.glob("**/*"))
            remaining_files = [f for f in remaining if f.is_file()]
            if not remaining_files:
                shutil.rmtree(str(item_path))
                print(f"Removed empty folder: {item_name}")

        print(f"Imported {len(imported)} PDFs from '{item_name}' to '{collection}'")
        return imported

    def extract_zip_to_inbox(self, zip_name: str) -> List[str]:
        """
        Extract a ZIP file in the inbox to a subfolder for review.
        Does NOT move to collection - just unpacks for inspection.

        Args:
            zip_name: Name of ZIP file in inbox

        Returns:
            List of extracted PDF paths
        """
        zip_path = self.inbox_path / zip_name
        if not zip_path.exists() or zip_path.suffix.lower() != '.zip':
            print(f"ZIP not found: {zip_name}")
            return []

        # Create folder with same name as ZIP (minus extension)
        folder_name = zip_path.stem
        extract_path = self.inbox_path / folder_name
        extract_path.mkdir(exist_ok=True)

        extracted = []
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.lower().endswith('.pdf'):
                    filename = Path(name).name
                    dest_path = extract_path / filename

                    # Handle duplicates
                    if dest_path.exists():
                        base = dest_path.stem
                        suffix = dest_path.suffix
                        counter = 1
                        while dest_path.exists():
                            dest_path = extract_path / f"{base}_{counter}{suffix}"
                            counter += 1

                    with zf.open(name) as src, open(dest_path, 'wb') as dst:
                        dst.write(src.read())

                    extracted.append(str(dest_path))

        print(f"Extracted {len(extracted)} PDFs from {zip_name} to {folder_name}/")

        # Optionally delete the ZIP after extraction
        # zip_path.unlink()

        return extracted

    def add_pdf(self, pdf_path: str, collection: str = None,
                move: bool = False) -> Optional[str]:
        """
        Add a PDF to a collection.

        Args:
            pdf_path: Path to PDF file
            collection: Collection ID to add to (required)
            move: If True, move file instead of copy

        Returns:
            Destination path or None if failed
        """
        if not collection:
            print("Error: collection is required")
            return None

        src_path = Path(pdf_path)
        if not src_path.exists():
            print(f"File not found: {pdf_path}")
            return None

        if not src_path.suffix.lower() == '.pdf':
            print(f"Not a PDF: {pdf_path}")
            return None

        # Ensure collection exists
        collection_path = self._get_collection_path(collection)
        collection_path.mkdir(parents=True, exist_ok=True)

        # Destination path
        dest_path = collection_path / src_path.name

        # Handle duplicates
        if dest_path.exists():
            # Add number suffix
            base = dest_path.stem
            suffix = dest_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = collection_path / f"{base}_{counter}{suffix}"
                counter += 1

        # Copy or move
        if move:
            shutil.move(str(src_path), str(dest_path))
            print(f"Moved: {src_path.name} -> {collection}/")
        else:
            shutil.copy2(str(src_path), str(dest_path))
            print(f"Copied: {src_path.name} -> {collection}/")

        return str(dest_path)

    def add_folder(self, folder_path: str, collection: str = "uncategorized",
                   recursive: bool = True, move: bool = False) -> List[str]:
        """
        Add all PDFs from a folder to a collection.

        Args:
            folder_path: Path to folder containing PDFs
            collection: Collection ID to add to
            recursive: Search subdirectories
            move: If True, move files instead of copy

        Returns:
            List of added file paths
        """
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Folder not found: {folder_path}")
            return []

        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(folder.glob(pattern))

        print(f"Found {len(pdf_files)} PDFs in {folder_path}")

        added = []
        for pdf in pdf_files:
            result = self.add_pdf(str(pdf), collection, move=move)
            if result:
                added.append(result)

        return added

    def add_zip(self, zip_path: str, collection: str = "uncategorized") -> List[str]:
        """
        Extract PDFs from a ZIP file into a collection.

        Args:
            zip_path: Path to ZIP file
            collection: Collection ID to add to

        Returns:
            List of extracted file paths
        """
        zip_file = Path(zip_path)
        if not zip_file.exists():
            print(f"ZIP not found: {zip_path}")
            return []

        collection_path = self._get_collection_path(collection)
        collection_path.mkdir(parents=True, exist_ok=True)

        extracted = []
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.lower().endswith('.pdf'):
                    # Extract to collection folder
                    # Use just the filename, not nested paths
                    filename = Path(name).name
                    dest_path = collection_path / filename

                    # Handle duplicates
                    if dest_path.exists():
                        base = dest_path.stem
                        suffix = dest_path.suffix
                        counter = 1
                        while dest_path.exists():
                            dest_path = collection_path / f"{base}_{counter}{suffix}"
                            counter += 1

                    # Extract
                    with zf.open(name) as src, open(dest_path, 'wb') as dst:
                        dst.write(src.read())

                    print(f"Extracted: {filename}")
                    extracted.append(str(dest_path))

        print(f"Extracted {len(extracted)} PDFs from {zip_file.name}")
        return extracted

    def detect_doi(self, text: str) -> Optional[str]:
        """
        Detect DOI in text content.

        Args:
            text: Text to search for DOI

        Returns:
            DOI string or None
        """
        # DOI pattern: 10.XXXX/anything
        doi_pattern = r'10\.\d{4,}/[^\s\]>)"}]+'
        match = re.search(doi_pattern, text)
        if match:
            doi = match.group(0)
            # Clean up trailing punctuation
            doi = doi.rstrip('.,;:')
            return doi
        return None

    def lookup_crossref(self, doi: str) -> Optional[Dict]:
        """
        Look up citation info from CrossRef API.

        Args:
            doi: DOI to look up

        Returns:
            Citation metadata or None
        """
        if not HAS_REQUESTS:
            return None

        try:
            url = f"https://api.crossref.org/works/{doi}"
            headers = {"User-Agent": "DisasterClippy/1.0 (mailto:admin@example.com)"}

            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                message = data.get('message', {})

                # Extract useful fields
                return {
                    'title': message.get('title', [''])[0],
                    'authors': [
                        f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in message.get('author', [])
                    ],
                    'publisher': message.get('publisher', ''),
                    'published_date': '-'.join(str(x) for x in message.get('published-print', {}).get('date-parts', [[]])[0]),
                    'type': message.get('type', ''),
                    'url': message.get('URL', ''),
                    'license': message.get('license', [{}])[0].get('URL', '') if message.get('license') else ''
                }
        except Exception as e:
            print(f"CrossRef lookup failed for {doi}: {e}")

        return None

    def classify_license(self, license_text: str) -> str:
        """
        Classify a license string into a category.

        Args:
            license_text: License text or URL

        Returns:
            License category: public_domain, open_access, restricted, unknown
        """
        if not license_text:
            return "unknown"

        text_lower = license_text.lower()

        # Public domain indicators
        if any(x in text_lower for x in ['public domain', 'cc0', 'government work', 'pd']):
            return "public_domain"

        # Open access indicators
        if any(x in text_lower for x in ['cc-by', 'cc by', 'creative commons', 'open access', 'cc-sa', 'attribution']):
            return "open_access"

        # Restricted indicators
        if any(x in text_lower for x in ['all rights reserved', 'copyright', 'restricted', 'proprietary']):
            return "restricted"

        return "unknown"

    def can_host_publicly(self, license_category: str) -> bool:
        """Check if content with this license can be hosted publicly on R2"""
        return LICENSE_CATEGORIES.get(license_category, {}).get('can_host', False)


def create_collection_manager(inbox_path: str = None, collections_path: str = None) -> PDFCollectionManager:
    """Convenience function to create a collection manager"""
    return PDFCollectionManager(inbox_path=inbox_path, collections_path=collections_path)


# Quick test
if __name__ == "__main__":
    manager = PDFCollectionManager()

    print(f"PDF Inbox at: {manager.inbox_path}")
    print(f"PDF Collections at: {manager.collections_path}")

    # Show inbox contents
    print("\n--- INBOX (needs sorting) ---")
    inbox = manager.list_inbox()
    if inbox['files']:
        print(f"  Files: {len(inbox['files'])} PDFs")
        for f in inbox['files'][:5]:
            print(f"    - {f['name']}")
    if inbox['folders']:
        print(f"  Folders: {len(inbox['folders'])}")
        for f in inbox['folders']:
            print(f"    - {f['name']}/ ({f['pdf_count']} PDFs)")
    if inbox['zips']:
        print(f"  ZIPs: {len(inbox['zips'])}")
        for z in inbox['zips']:
            print(f"    - {z['name']} ({z['pdf_count']} PDFs inside)")
    if not any([inbox['files'], inbox['folders'], inbox['zips']]):
        print("  (empty)")

    # Show collections
    print("\n--- COLLECTIONS (organized) ---")
    collections = manager.list_collections()
    if collections:
        for coll in collections:
            print(f"  {coll['collection_id']}: {coll['pdf_count']} PDFs, {coll['document_count']} indexed")
    else:
        print("  (none)")
