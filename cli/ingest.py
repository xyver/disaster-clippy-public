#!/usr/bin/env python3
"""
Ingest script for Disaster Clippy.
Run this locally to scrape content and populate the vector database.

Supports ALL scraper types: mediawiki, static, pdf, fandom, etc.

Usage:
    # List available scrapers
    python -m cli.ingest list-scrapers

    # Scrape from any MediaWiki site
    python -m cli.ingest scrape mediawiki --url https://www.appropedia.org --search "solar"
    python -m cli.ingest scrape mediawiki --url https://www.appropedia.org --category "Water"

    # Scrape from preset sources
    python -m cli.ingest scrape appropedia --search "hexayurt" --limit 50
    python -m cli.ingest scrape builditsolar --limit 100

    # Scrape from Fandom wikis
    python -m cli.ingest scrape fandom --wiki solarcooking --search "box cooker"

    # Process PDFs
    python -m cli.ingest scrape pdf --file /path/to/document.pdf
    python -m cli.ingest scrape pdf --folder /path/to/pdfs --chunked

    # Sync from config file (for reproducible builds)
    python -m cli.ingest sync                  # Run all sources from config
    python -m cli.ingest sync --clear          # Clear DB first, then sync

    # Database management
    python -m cli.ingest stats                 # Show database statistics
    python -m cli.ingest index                 # Show metadata index (fast!)
    python -m cli.ingest rebuild-index         # Rebuild metadata index from DB
    python -m cli.ingest clear                 # Clear all data (use with caution!)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

from offline_tools.scraper import get_scraper, list_scrapers, SCRAPER_REGISTRY
from offline_tools.scraper.pdf_collections import PDFCollectionManager
from offline_tools.vectordb import VectorStore, MetadataIndex

# Config file path (no longer used, kept for reference)
CONFIG_FILE = Path(__file__).parent.parent / "local_settings.json"


def ingest_documents(documents: List[dict], smart: bool = True) -> int:
    """
    Add documents to vector store with optional smart deduplication.

    Args:
        documents: List of document dicts (from ScrapedPage.to_dict())
        smart: If True, skip documents already in metadata index

    Returns:
        Number of documents added
    """
    if not documents:
        return 0

    store = VectorStore()

    if smart:
        existing_titles = store.metadata_index.get_titles()
        new_docs = []
        for doc in documents:
            if doc.get("title") not in existing_titles:
                new_docs.append(doc)
        skipped = len(documents) - len(new_docs)
        if skipped > 0:
            print(f"  Skipping {skipped} documents already in database")
        documents = new_docs

    if not documents:
        print("All documents already in database - nothing to add!")
        return 0

    print(f"\nComputing embeddings and adding {len(documents)} documents...")
    count = store.add_documents(documents)
    print(f"Added {count} documents to the database")
    print(f"Total documents: {store.get_stats()['total_documents']}")
    return count


def cmd_scrape(args):
    """Handle the scrape command for any scraper type"""
    scraper_type = args.scraper_type.lower()

    # Build scraper kwargs based on type
    kwargs = {"rate_limit": args.rate_limit}

    if scraper_type == "mediawiki":
        if not args.url:
            print("Error: --url required for mediawiki scraper")
            return 1
        kwargs["base_url"] = args.url
        if args.source_name:
            kwargs["source_name"] = args.source_name

    elif scraper_type == "fandom":
        if not args.wiki:
            print("Error: --wiki required for fandom scraper (e.g., --wiki solarcooking)")
            return 1
        kwargs["wiki_name"] = args.wiki

    elif scraper_type == "static":
        if not args.url:
            print("Error: --url required for static scraper")
            return 1
        kwargs["base_url"] = args.url
        if args.source_name:
            kwargs["source_name"] = args.source_name

    elif scraper_type == "pdf":
        if args.source_name:
            kwargs["source_name"] = args.source_name
        else:
            kwargs["source_name"] = "pdf"

    elif scraper_type == "substack":
        if not args.csv:
            print("Error: --csv required for substack scraper (path to posts.csv)")
            return 1
        if not args.url:
            print("Error: --url required for substack scraper (newsletter URL)")
            return 1
        kwargs["csv_path"] = args.csv
        kwargs["newsletter_url"] = args.url
        if args.source_name:
            kwargs["source_name"] = args.source_name

    elif scraper_type in ["appropedia", "builditsolar"]:
        # Preset scrapers need no extra config
        pass

    else:
        print(f"Unknown scraper type: {scraper_type}")
        print(f"Available: {', '.join(list_scrapers())}")
        return 1

    # Create scraper
    try:
        scraper = get_scraper(scraper_type, **kwargs)
        print(f"Created {scraper_type} scraper")
    except Exception as e:
        print(f"Error creating scraper: {e}")
        return 1

    # Handle PDF scraper separately (different interface)
    if scraper_type == "pdf":
        return _handle_pdf_scrape(scraper, args)

    # Handle web scrapers
    return _handle_web_scrape(scraper, args)


def _handle_web_scrape(scraper, args) -> int:
    """Handle scraping for web-based scrapers (mediawiki, static, etc.)"""
    pages = []
    limit = args.limit

    # Search mode
    if args.search:
        print(f"Searching for: '{args.search}' (limit: {limit})...")
        if hasattr(scraper, 'search_pages'):
            urls = scraper.search_pages(args.search, limit=limit)
        else:
            print("Error: This scraper doesn't support search. Use --all instead.")
            return 1

        print(f"Found {len(urls)} pages")
        for i, url in enumerate(urls):
            print(f"  [{i+1}/{len(urls)}] Scraping: {url}")
            page = scraper.scrape_page(url)
            if page:
                pages.append(page)

    # Category mode (MediaWiki only)
    elif args.category:
        print(f"Fetching category: '{args.category}' (limit: {limit})...")
        if hasattr(scraper, 'get_category_pages'):
            urls = scraper.get_category_pages(args.category, limit=limit)
        elif hasattr(scraper, 'get_page_list'):
            # Try passing category to get_page_list for Appropedia
            urls = scraper.get_page_list(limit=limit, categories=[args.category])
        else:
            print("Error: This scraper doesn't support categories.")
            return 1

        print(f"Found {len(urls)} pages")
        for i, url in enumerate(urls):
            print(f"  [{i+1}/{len(urls)}] Scraping: {url}")
            page = scraper.scrape_page(url)
            if page:
                pages.append(page)

    # All pages mode
    elif args.all:
        print(f"Fetching all pages (limit: {limit})...")
        urls = scraper.get_page_list(limit=limit)
        print(f"Found {len(urls)} pages")
        for i, url in enumerate(urls):
            print(f"  [{i+1}/{len(urls)}] Scraping: {url}")
            page = scraper.scrape_page(url)
            if page:
                pages.append(page)

    else:
        print("Error: Specify --search, --category, or --all")
        return 1

    print(f"\nSuccessfully scraped {len(pages)} pages")

    if not pages:
        return 0

    # Add to vector store
    documents = [page.to_dict() for page in pages]
    ingest_documents(documents, smart=not args.force)
    return 0


def _handle_pdf_scrape(scraper, args) -> int:
    """Handle PDF scraping"""
    pages = []

    if args.file:
        # Single file
        path = Path(args.file)
        if not path.exists():
            print(f"Error: File not found: {args.file}")
            return 1

        print(f"Processing PDF: {path.name}")
        if args.chunked:
            pages = scraper.process_file_chunked(
                str(path),
                chunk_size=args.chunk_size,
                overlap=args.overlap
            )
        else:
            page = scraper.process_file(str(path))
            if page:
                pages = [page]

    elif args.folder:
        # Folder of PDFs
        folder = Path(args.folder)
        if not folder.exists():
            print(f"Error: Folder not found: {args.folder}")
            return 1

        pdf_files = list(folder.glob("**/*.pdf" if args.recursive else "*.pdf"))
        print(f"Found {len(pdf_files)} PDFs in {folder}")

        for i, pdf_path in enumerate(pdf_files[:args.limit] if args.limit else pdf_files):
            print(f"  [{i+1}/{len(pdf_files)}] Processing: {pdf_path.name}")
            if args.chunked:
                chunks = scraper.process_file_chunked(
                    str(pdf_path),
                    chunk_size=args.chunk_size,
                    overlap=args.overlap
                )
                pages.extend(chunks)
            else:
                page = scraper.process_file(str(pdf_path))
                if page:
                    pages.append(page)

    elif args.url:
        # PDF from URL
        print(f"Downloading PDF from: {args.url}")
        page = scraper.process_url(args.url)
        if page:
            pages = [page]

    else:
        print("Error: Specify --file, --folder, or --url for PDF scraper")
        return 1

    print(f"\nSuccessfully processed {len(pages)} documents")

    if not pages:
        return 0

    # Add to vector store
    documents = [page.to_dict() for page in pages]
    ingest_documents(documents, smart=not args.force)
    return 0


def cmd_list_scrapers(args):
    """List available scraper types"""
    print("Available scraper types:")
    print()
    for name, cls in SCRAPER_REGISTRY.items():
        doc = cls.__doc__ or "No description"
        first_line = doc.strip().split('\n')[0]
        print(f"  {name:15} - {first_line}")
    print()
    print("Examples:")
    print("  python -m cli.ingest scrape appropedia --search 'solar cooker'")
    print("  python -m cli.ingest scrape mediawiki --url https://wiki.example.com --all")
    print("  python -m cli.ingest scrape fandom --wiki solarcooking --search 'box'")
    print("  python -m cli.ingest scrape pdf --file document.pdf --chunked")
    print("  python -m cli.ingest scrape substack --csv posts.csv --url https://example.substack.com --all")


def cmd_sync(args):
    """
    Run all enabled sources from config file.
    This ensures local and production databases have the same content.
    """
    if not CONFIG_FILE.exists():
        print(f"Config file not found: {CONFIG_FILE}")
        print("Create config/ingest_config.json with your sources.")
        return 1

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    sources = config.get("sources", [])
    enabled_sources = [s for s in sources if s.get("enabled", True)]

    if not enabled_sources:
        print("No enabled sources in config.")
        return 1

    print(f"Found {len(enabled_sources)} enabled source(s) in config")
    print("=" * 50)

    if args.clear:
        print("\nClearing existing database...")
        store = VectorStore()
        store.delete_all()
        print("Database cleared.\n")

    for i, source in enumerate(enabled_sources, 1):
        source_type = source.get("type", source.get("scraper_type", "mediawiki"))
        print(f"\n[{i}/{len(enabled_sources)}] Processing: {source.get('name', source_type)}")
        print("-" * 40)

        try:
            # Build kwargs from source config
            kwargs = {"rate_limit": source.get("rate_limit", 1.0)}

            if source.get("base_url"):
                kwargs["base_url"] = source["base_url"]
            if source.get("wiki_name"):
                kwargs["wiki_name"] = source["wiki_name"]
            if source.get("source_name"):
                kwargs["source_name"] = source["source_name"]

            scraper = get_scraper(source_type, **kwargs)

            # Get pages based on source config
            limit = source.get("limit", 100)
            pages = []

            if source.get("search"):
                urls = scraper.search_pages(source["search"], limit=limit)
                for url in urls:
                    page = scraper.scrape_page(url)
                    if page:
                        pages.append(page)

            elif source.get("category"):
                if hasattr(scraper, 'get_category_pages'):
                    urls = scraper.get_category_pages(source["category"], limit=limit)
                else:
                    urls = scraper.get_page_list(limit=limit, categories=[source["category"]])
                for url in urls:
                    page = scraper.scrape_page(url)
                    if page:
                        pages.append(page)

            elif source.get("all", False):
                urls = scraper.get_page_list(limit=limit)
                for url in urls:
                    page = scraper.scrape_page(url)
                    if page:
                        pages.append(page)

            print(f"Scraped {len(pages)} pages")
            if pages:
                documents = [p.to_dict() for p in pages]
                ingest_documents(documents, smart=True)

        except Exception as e:
            print(f"Error processing source: {e}")
            continue

    print("\n" + "=" * 50)
    print("Sync complete!")
    cmd_stats(args)
    return 0


def cmd_stats(args):
    """Show database statistics"""
    store = VectorStore()
    stats = store.get_stats()
    print(f"Database Statistics:")
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Total documents: {stats['total_documents']}")


def cmd_index(args):
    """Show metadata index summary (fast - no DB query)"""
    index = MetadataIndex()
    print(index.export_summary())


def cmd_rebuild_index(args):
    """Rebuild metadata index from vector database"""
    store = VectorStore()
    store.rebuild_metadata_index()
    print("\nIndex rebuilt. Summary:")
    print(store.metadata_index.export_summary())


def cmd_clear(args):
    """Clear all documents from the database"""
    confirm = input("Are you sure you want to delete ALL documents? (yes/no): ")
    if confirm.lower() != "yes":
        print("Aborted.")
        return 0

    store = VectorStore()
    store.delete_all()
    print("Database cleared.")
    return 0


# ============================================================================
# PDF Collection Commands
# ============================================================================

def cmd_pdf(args):
    """Handle PDF collection subcommands"""
    if args.pdf_command == "inbox":
        return cmd_pdf_inbox(args)
    elif args.pdf_command == "list":
        return cmd_pdf_list(args)
    elif args.pdf_command == "create":
        return cmd_pdf_create(args)
    elif args.pdf_command == "import":
        return cmd_pdf_import(args)
    elif args.pdf_command == "add":
        return cmd_pdf_add(args)
    elif args.pdf_command == "process":
        return cmd_pdf_process(args)
    else:
        print("Unknown PDF command. Use: inbox, list, create, import, add, process")
        return 1


def cmd_pdf_inbox(args):
    """Show inbox contents (unsorted PDFs)"""
    manager = PDFCollectionManager()
    inbox = manager.list_inbox()

    print(f"PDF Inbox ({manager.inbox_path}):\n")

    total_items = len(inbox['files']) + len(inbox['folders']) + len(inbox['zips'])
    if total_items == 0:
        print("  (empty - drop PDFs, folders, or ZIPs here)")
        return 0

    if inbox['files']:
        print(f"Files ({len(inbox['files'])} PDFs):")
        for f in inbox['files']:
            size_kb = f['size'] // 1024
            print(f"  - {f['name']} ({size_kb} KB)")

    if inbox['folders']:
        print(f"\nFolders ({len(inbox['folders'])}):")
        for f in inbox['folders']:
            print(f"  - {f['name']}/ ({f['pdf_count']} PDFs)")

    if inbox['zips']:
        print(f"\nZIP Archives ({len(inbox['zips'])}):")
        for z in inbox['zips']:
            size_mb = z['size'] // (1024 * 1024)
            print(f"  - {z['name']} ({z['pdf_count']} PDFs, {size_mb} MB)")

    print(f"\nUse 'pdf import <item> --collection <name>' to move to a collection")
    return 0


def cmd_pdf_list(args):
    """List PDF collections"""
    manager = PDFCollectionManager()
    collections = manager.list_collections()

    print(f"PDF Collections ({manager.collections_path}):\n")

    if not collections:
        print("  (none - use 'pdf create <name>' to create one)")
        return 0

    print(f"{'Collection':<25} {'PDFs':<8} {'Indexed':<10} {'License':<20}")
    print("-" * 65)

    for coll in collections:
        print(f"{coll['collection_id']:<25} {coll['pdf_count']:<8} {coll['document_count']:<10} {coll['license']:<20}")

    return 0


def cmd_pdf_create(args):
    """Create a new PDF collection"""
    manager = PDFCollectionManager()

    topics = args.topics.split(",") if args.topics else []

    coll = manager.create_collection(
        collection_id=args.name,
        name=args.display_name or args.name.replace("_", " ").replace("-", " ").title(),
        description=args.description or "",
        license=args.license or "Unknown",
        topics=topics
    )

    print(f"\nCreated collection: {coll.collection_id}")
    print(f"  Name: {coll.name}")
    print(f"  License: {coll.license}")
    print(f"  Topics: {', '.join(coll.topics) if coll.topics else 'None'}")
    print(f"  Path: {manager._get_collection_path(coll.collection_id)}")

    return 0


def cmd_pdf_import(args):
    """Import item from inbox into a collection"""
    manager = PDFCollectionManager()

    item_name = args.item
    collection = args.collection

    if not collection:
        print("Error: --collection is required")
        print("Available collections:")
        for coll in manager.list_collections():
            print(f"  - {coll['collection_id']}")
        print("\nOr create a new one with: pdf create <name>")
        return 1

    # Check if collection exists, offer to create
    existing = [c['collection_id'] for c in manager.list_collections()]
    if collection not in existing:
        print(f"Collection '{collection}' doesn't exist. Creating it...")

    # Import the item
    imported = manager.import_from_inbox(item_name, collection)

    if imported:
        print(f"\nSuccessfully imported {len(imported)} PDF(s) to '{collection}'")

        # Show collection status
        for coll in manager.list_collections():
            if coll['collection_id'] == collection:
                print(f"Collection now has {coll['pdf_count']} PDFs")
                break
    else:
        print(f"No PDFs imported from '{item_name}'")

    return 0


def cmd_pdf_add(args):
    """Add PDFs directly to a collection (bypassing inbox)"""
    manager = PDFCollectionManager()
    collection = args.collection

    added = []

    if args.file:
        result = manager.add_pdf(args.file, collection=collection, move=args.move)
        if result:
            added.append(result)

    elif args.folder:
        added = manager.add_folder(args.folder, collection=collection,
                                   recursive=not args.no_recursive, move=args.move)

    elif args.zip:
        added = manager.add_zip(args.zip, collection=collection)

    else:
        print("Error: Specify --file, --folder, or --zip")
        return 1

    print(f"\nAdded {len(added)} PDF(s) to '{collection}'")

    # Show updated collection info
    collections = manager.list_collections()
    for coll in collections:
        if coll['collection_id'] == collection:
            print(f"Collection now has {coll['pdf_count']} PDFs")
            break

    return 0


def cmd_pdf_process(args):
    """Process a collection and add to vector database"""
    import hashlib
    import os
    from offline_tools.scraper.pdf import PDFScraper, HAS_PYMUPDF, HAS_PYPDF
    from datetime import datetime

    manager = PDFCollectionManager()
    collection_id = args.collection

    # Check collection exists
    collection_path = manager._get_collection_path(collection_id)
    if not collection_path.exists():
        print(f"Collection not found: {collection_id}")
        return 1

    # Load collection metadata
    coll_meta = manager._load_collection_metadata(collection_id)
    if not coll_meta:
        print(f"Warning: No _collection.json found, using defaults")
        source_name = f"pdf_{collection_id}"
    else:
        source_name = f"pdf_{collection_id}"
        print(f"Processing collection: {coll_meta.name}")
        print(f"  License: {coll_meta.license}")
        print(f"  Topics: {', '.join(coll_meta.topics)}")

    # Find PDFs
    pdf_files = list(collection_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {collection_path}")
        return 1

    print(f"\nFound {len(pdf_files)} PDFs to process")

    # Create scraper
    scraper = PDFScraper(source_name=source_name)

    # Helper to extract rich PDF metadata
    def extract_rich_metadata(pdf_path, pages):
        """Extract all available metadata from PDF for individual document tracking"""
        try:
            # Get raw PDF metadata
            pdf_meta = scraper._extract_metadata(str(pdf_path))

            # Get page count
            page_count = None
            if HAS_PYMUPDF:
                import fitz
                doc = fitz.open(str(pdf_path))
                page_count = len(doc)
                doc.close()
            elif HAS_PYPDF:
                from pypdf import PdfReader
                reader = PdfReader(str(pdf_path))
                page_count = len(reader.pages)

            # Compute content hash
            content_sample = pages[0].content[:1000] if pages else ""
            content_hash = hashlib.md5(content_sample.encode()).hexdigest()[:12]

            # Parse title - clean up if it looks like garbage
            title = pdf_meta.get("title", "") or ""
            if not title or len(title) < 3 or title.startswith("Microsoft"):
                title = pages[0].title if pages else pdf_path.stem

            # Parse authors
            authors = pdf_meta.get("author", "") or ""
            if authors:
                # Split on common separators
                import re
                authors = [a.strip() for a in re.split(r'[,;]|and|\n', authors) if a.strip()]
            else:
                authors = []

            # Parse creation date
            creation_date = pdf_meta.get("creation_date", "")
            if creation_date:
                # Try to parse PDF date format (D:YYYYMMDDHHmmss)
                import re
                date_match = re.search(r'D:(\d{4})(\d{2})?(\d{2})?', str(creation_date))
                if date_match:
                    year = date_match.group(1)
                    month = date_match.group(2) or "01"
                    day = date_match.group(3) or "01"
                    creation_date = f"{year}-{month}-{day}"
                elif len(str(creation_date)) == 4 and creation_date.isdigit():
                    creation_date = creation_date  # Just year
                else:
                    creation_date = ""

            # Extract keywords/tags from PDF and content
            keywords = pdf_meta.get("keywords", "") or ""
            tags = []
            if keywords:
                import re
                tags = [k.strip() for k in re.split(r'[,;]', keywords) if k.strip()]
            # Also get any categories the scraper found
            if pages and pages[0].categories:
                tags = list(set(tags + pages[0].categories))

            # Get file size
            file_size = os.path.getsize(str(pdf_path))

            return {
                "filename": pdf_path.name,
                "title": title,
                "authors": authors if authors else None,
                "publication_date": creation_date if creation_date else None,
                "tags": tags[:15] if tags else None,  # Cap at 15 tags
                "content_hash": content_hash,
                "char_count": sum(len(p.content) for p in pages),
                "chunk_count": len(pages),
                "page_count": page_count,
                "file_size": file_size,
                "subject": pdf_meta.get("subject") if pdf_meta.get("subject") else None,
                "creator": pdf_meta.get("creator") if pdf_meta.get("creator") else None,
                "added_at": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"    Warning: Could not extract full metadata: {e}")
            # Return minimal metadata
            content_sample = pages[0].content[:1000] if pages else ""
            content_hash = hashlib.md5(content_sample.encode()).hexdigest()[:12]
            return {
                "filename": pdf_path.name,
                "title": pages[0].title if pages else pdf_path.stem,
                "content_hash": content_hash,
                "char_count": sum(len(p.content) for p in pages),
                "chunk_count": len(pages),
                "added_at": datetime.now().isoformat()
            }

    # Process each PDF and track document metadata
    all_pages = []
    pdf_doc_info = {}  # hash -> rich metadata
    all_doc_tags = set()  # Collect all tags for rollup

    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")

        if args.chunked:
            pages = scraper.process_file_chunked(
                str(pdf_path),
                chunk_size=args.chunk_size,
                overlap=args.overlap
            )
        else:
            page = scraper.process_file(str(pdf_path))
            pages = [page] if page else []

        if pages:
            # Extract rich metadata for this document
            doc_info = extract_rich_metadata(pdf_path, pages)
            pdf_doc_info[doc_info['content_hash']] = doc_info

            # Collect tags for rollup to collection level
            if doc_info.get('tags'):
                all_doc_tags.update(doc_info['tags'])

        all_pages.extend(pages)

    print(f"\nExtracted {len(all_pages)} document chunks")

    if not all_pages:
        print("No content extracted from PDFs")
        return 1

    # Add categories from collection topics
    if coll_meta and coll_meta.topics:
        for page in all_pages:
            page.categories = list(set(page.categories + coll_meta.topics))

    # Convert to dicts and ingest
    documents = [p.to_dict() for p in all_pages]
    count = ingest_documents(documents, smart=not args.force)

    # Update collection manifest with document metadata
    if pdf_doc_info:
        manifest_path = manager._get_collection_manifest_path(collection_id)
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            # Update documents section with rich metadata
            if 'documents' not in manifest:
                manifest['documents'] = {}

            for content_hash, doc_info in pdf_doc_info.items():
                # Clean None values for cleaner JSON
                clean_info = {k: v for k, v in doc_info.items() if v is not None}
                manifest['documents'][content_hash] = clean_info

            # Roll up ALL document tags to collection topics for discoverability
            existing_topics = set(manifest.get('collection', {}).get('topics', []))
            if all_doc_tags:
                merged_topics = list(existing_topics | all_doc_tags)
                if len(merged_topics) > 25:
                    from collections import Counter
                    tag_counts = Counter()
                    for doc_info in pdf_doc_info.values():
                        if doc_info.get('tags'):
                            tag_counts.update(doc_info['tags'])
                    new_tags = all_doc_tags - existing_topics
                    top_new = [tag for tag, _ in tag_counts.most_common() if tag in new_tags]
                    room_for_new = 25 - len(existing_topics)
                    merged_topics = list(existing_topics) + top_new[:room_for_new]
                manifest['collection']['topics'] = merged_topics
                new_count = len(set(merged_topics) - existing_topics)
                if new_count:
                    print(f"  Added {new_count} tags to collection topics (total: {len(merged_topics)})")

            # Update timestamp
            manifest['collection']['updated'] = datetime.now().isoformat()

            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            print(f"Updated collection manifest with {len(pdf_doc_info)} document(s)")

    print(f"\nProcessed {len(pdf_files)} PDFs -> {count} documents indexed")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Ingest content into Disaster Clippy's vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m cli.ingest list-scrapers
    python -m cli.ingest scrape appropedia --search "water filter" --limit 50
    python -m cli.ingest scrape mediawiki --url https://wiki.example.com --all
    python -m cli.ingest scrape fandom --wiki solarcooking --search "parabolic"
    python -m cli.ingest scrape pdf --folder ./pdfs --chunked
    python -m cli.ingest sync
    python -m cli.ingest stats
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list-scrapers command
    subparsers.add_parser("list-scrapers", help="List available scraper types")

    # scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape content from a source")
    scrape_parser.add_argument("scraper_type", help="Scraper type (e.g., mediawiki, pdf, fandom)")

    # Source configuration
    scrape_parser.add_argument("--url", help="Base URL for mediawiki/static/substack scrapers")
    scrape_parser.add_argument("--wiki", help="Wiki name for fandom scraper (e.g., solarcooking)")
    scrape_parser.add_argument("--csv", help="CSV file path for substack scraper (posts.csv)")
    scrape_parser.add_argument("--source-name", help="Custom source name for metadata")

    # What to scrape
    scrape_parser.add_argument("--search", help="Search query")
    scrape_parser.add_argument("--category", help="Category name (MediaWiki)")
    scrape_parser.add_argument("--all", action="store_true", help="Scrape all pages from sitemap/API")

    # PDF-specific options
    scrape_parser.add_argument("--file", help="Single PDF file path")
    scrape_parser.add_argument("--folder", help="Folder containing PDFs")
    scrape_parser.add_argument("--recursive", action="store_true", help="Search subfolders for PDFs")
    scrape_parser.add_argument("--chunked", action="store_true", help="Split large PDFs into chunks")
    scrape_parser.add_argument("--chunk-size", type=int, default=4000, help="Chunk size in chars (default: 4000)")
    scrape_parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks (default: 200)")

    # Common options
    scrape_parser.add_argument("--limit", type=int, default=100, help="Max pages to scrape (default: 100)")
    scrape_parser.add_argument("--rate-limit", type=float, default=1.0, help="Seconds between requests (default: 1.0)")
    scrape_parser.add_argument("--force", action="store_true", help="Don't skip existing documents")

    # sync command
    sync_parser = subparsers.add_parser("sync", help="Run all sources from config file")
    sync_parser.add_argument("--clear", action="store_true", help="Clear database before syncing")

    # stats command
    subparsers.add_parser("stats", help="Show database statistics")

    # index command
    subparsers.add_parser("index", help="Show metadata index summary (fast, no DB query)")

    # rebuild-index command
    subparsers.add_parser("rebuild-index", help="Rebuild metadata index from vector database")

    # clear command
    subparsers.add_parser("clear", help="Clear all data from database")

    # pdf command with subcommands
    pdf_parser = subparsers.add_parser("pdf", help="PDF collection management")
    pdf_subparsers = pdf_parser.add_subparsers(dest="pdf_command", help="PDF subcommand")

    # pdf inbox - show what's waiting to be sorted
    pdf_subparsers.add_parser("inbox", help="Show inbox contents (unsorted PDFs)")

    # pdf list - show organized collections
    pdf_subparsers.add_parser("list", help="List PDF collections")

    # pdf create - create a new collection
    pdf_create = pdf_subparsers.add_parser("create", help="Create a new PDF collection")
    pdf_create.add_argument("name", help="Collection ID (folder name)")
    pdf_create.add_argument("--display-name", help="Display name for collection")
    pdf_create.add_argument("--description", help="Collection description")
    pdf_create.add_argument("--license", help="License type (e.g., 'Public Domain', 'CC-BY')")
    pdf_create.add_argument("--topics", help="Comma-separated topic tags")

    # pdf import - move from inbox to collection
    pdf_import = pdf_subparsers.add_parser("import", help="Import from inbox to collection")
    pdf_import.add_argument("item", help="Item name in inbox (file, folder, or ZIP)")
    pdf_import.add_argument("--collection", "-c", required=True, help="Collection to import into")

    # pdf add - add external files directly to collection (bypass inbox)
    pdf_add = pdf_subparsers.add_parser("add", help="Add external PDFs directly to a collection")
    pdf_add.add_argument("--collection", "-c", required=True, help="Collection to add to")
    pdf_add.add_argument("--file", "-f", help="Single PDF file to add")
    pdf_add.add_argument("--folder", "-d", help="Folder of PDFs to add")
    pdf_add.add_argument("--zip", "-z", help="ZIP file containing PDFs")
    pdf_add.add_argument("--move", action="store_true", help="Move files instead of copy")
    pdf_add.add_argument("--no-recursive", action="store_true", help="Don't search subfolders")

    # pdf process
    pdf_process = pdf_subparsers.add_parser("process", help="Process collection and add to vector DB")
    pdf_process.add_argument("collection", help="Collection ID to process")
    pdf_process.add_argument("--chunked", action="store_true", help="Split large PDFs into chunks")
    pdf_process.add_argument("--chunk-size", type=int, default=4000, help="Chunk size in chars")
    pdf_process.add_argument("--overlap", type=int, default=200, help="Overlap between chunks")
    pdf_process.add_argument("--force", action="store_true", help="Re-process existing documents")

    args = parser.parse_args()

    if args.command == "list-scrapers":
        return cmd_list_scrapers(args)
    elif args.command == "scrape":
        return cmd_scrape(args)
    elif args.command == "sync":
        return cmd_sync(args)
    elif args.command == "stats":
        return cmd_stats(args)
    elif args.command == "index":
        return cmd_index(args)
    elif args.command == "rebuild-index":
        return cmd_rebuild_index(args)
    elif args.command == "clear":
        return cmd_clear(args)
    elif args.command == "pdf":
        return cmd_pdf(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
