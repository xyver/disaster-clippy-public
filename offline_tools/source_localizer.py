"""
Source Localization Engine for Phase 4.
Creates fully pre-translated source variants for native-language search and chat.

This module handles:
- Translating all HTML files in a source to the target language
- Generating native-language embeddings
- Creating language-specific ChromaDB instances
- Progress tracking with checkpointing for resume capability
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import hashlib


@dataclass
class LocalizationResult:
    """Result of a source localization operation"""
    success: bool
    source_id: str
    target_lang: str
    localized_source_id: str = ""
    documents_translated: int = 0
    documents_skipped: int = 0
    embeddings_generated: int = 0
    html_files_translated: int = 0
    resumed: bool = False
    checkpoint_progress: int = 0
    error: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class LocalizationCheckpoint:
    """Checkpoint for resumable localization jobs"""
    source_id: str
    target_lang: str
    progress: int = 0
    last_doc_index: int = 0
    documents_processed: int = 0
    html_files_processed: int = 0
    phase: str = "html"  # html, metadata, index, vectors, chromadb
    partial_files: Dict[str, str] = field(default_factory=dict)
    errors: List[Dict] = field(default_factory=list)
    created_at: str = ""
    last_saved: str = ""


def get_backup_path() -> Optional[Path]:
    """Get the backup folder path from config"""
    try:
        from admin.local_config import get_local_config
        config = get_local_config()
        backup_folder = config.get_backup_folder()
        if backup_folder:
            return Path(backup_folder)
    except Exception:
        pass

    # Fallback to env var
    backup_path = os.getenv("BACKUP_PATH")
    if backup_path:
        return Path(backup_path)

    return None


def localize_source(
    source_id: str,
    target_lang: str,
    progress_callback: Callable[[int, int, str], None] = None,
    cancel_checker: Callable[[], bool] = None,
    job_id: str = None,
    batch_size: int = 128,
    checkpoint_interval: int = 10,
    resume: bool = True,
    embedding_model: str = "all-mpnet-base-v2",
    force_overwrite: bool = False
) -> LocalizationResult:
    """
    Create a fully localized version of a source.

    This translates all HTML content, regenerates metadata with translated titles,
    creates a new index, generates native-language embeddings, and populates
    a language-specific ChromaDB instance.

    Args:
        source_id: Source to localize (e.g., "appropedia")
        target_lang: ISO 639-1 language code (e.g., "es", "fr")
        progress_callback: Function(current, total, message) for progress updates
        cancel_checker: Function() returns True if job should be cancelled
        job_id: Job ID for tracking (optional)
        batch_size: Documents per batch for translation/embedding
        checkpoint_interval: Save checkpoint every N documents
        resume: If True and checkpoint exists, resume from last position
        embedding_model: Model name for embeddings (e.g., "all-mpnet-base-v2", "all-MiniLM-L6-v2")
        force_overwrite: If True, delete existing localization and start fresh

    Returns:
        LocalizationResult with stats and any errors
    """
    result = LocalizationResult(
        success=False,
        source_id=source_id,
        target_lang=target_lang,
        localized_source_id=f"{source_id}_{target_lang}"
    )

    def update_progress(current: int, total: int, message: str):
        if progress_callback:
            progress_callback(current, total, message)

    def check_cancelled() -> bool:
        if cancel_checker and cancel_checker():
            return True
        return False

    try:
        # Get paths
        backup_path = get_backup_path()
        if not backup_path:
            result.error = "No backup folder configured"
            return result

        source_path = backup_path / source_id
        localized_path = backup_path / result.localized_source_id

        # Handle force_overwrite - delete existing localization if requested
        if force_overwrite and localized_path.exists():
            update_progress(0, 100, "Deleting existing localization (force overwrite)...")
            try:
                shutil.rmtree(localized_path)
                # Also delete any checkpoint
                _delete_checkpoint(source_id, target_lang)
            except Exception as e:
                result.error = f"Failed to delete existing localization: {e}"
                return result

        # Validate source exists
        if not source_path.exists():
            result.error = f"Source not found: {source_id}"
            return result

        # Check for required files
        metadata_file = source_path / "_metadata.json"
        manifest_file = source_path / "_manifest.json"
        pages_dir = source_path / "pages"

        if not metadata_file.exists():
            result.error = f"Metadata file not found: {metadata_file}"
            return result

        # Check translation service availability
        from offline_tools.translation import TranslationService
        translator = TranslationService(target_lang)
        if not translator.is_available():
            result.error = f"Translation model not installed for language: {target_lang}"
            return result

        update_progress(0, 100, f"Starting localization to {target_lang}...")

        # Load checkpoint if resuming
        checkpoint = None
        if resume:
            checkpoint = _load_checkpoint(source_id, target_lang)
            if checkpoint:
                result.resumed = True
                result.checkpoint_progress = checkpoint.progress
                update_progress(checkpoint.progress, 100, f"Resuming from {checkpoint.phase} phase...")

        # Create localized directory
        localized_path.mkdir(parents=True, exist_ok=True)
        localized_pages_dir = localized_path / "pages"
        localized_pages_dir.mkdir(parents=True, exist_ok=True)

        # Copy supporting files from parent source for self-contained localization
        zim_path_copied = None
        original_manifest = {}
        if manifest_file.exists():
            with open(manifest_file, 'r', encoding='utf-8') as f:
                original_manifest = json.load(f)

        # Copy backup_manifest.json (URL to filename mapping)
        backup_manifest_src = source_path / "backup_manifest.json"
        if backup_manifest_src.exists():
            backup_manifest_dest = localized_path / "backup_manifest.json"
            if not backup_manifest_dest.exists():
                shutil.copy2(backup_manifest_src, backup_manifest_dest)

        # Copy _zim_url_index.json (ZIM URL lookup cache - speeds up asset serving)
        zim_index_src = source_path / "_zim_url_index.json"
        if zim_index_src.exists():
            zim_index_dest = localized_path / "_zim_url_index.json"
            if not zim_index_dest.exists():
                shutil.copy2(zim_index_src, zim_index_dest)

        # Copy assets folder (for non-ZIM HTML sources with extracted images/CSS)
        assets_src = source_path / "assets"
        if assets_src.exists() and assets_src.is_dir():
            assets_dest = localized_path / "assets"
            if not assets_dest.exists():
                update_progress(1, 100, "Copying assets folder...")
                shutil.copytree(assets_src, assets_dest)

        # Copy ZIM file if parent is a ZIM source
        if original_manifest.get("created_from") == "zim_import":
            parent_zim_path = original_manifest.get("zim_path")
            if parent_zim_path and Path(parent_zim_path).exists():
                zim_filename = Path(parent_zim_path).name
                dest_zim_path = localized_path / zim_filename

                # Only copy if not already present (resume support)
                if not dest_zim_path.exists():
                    update_progress(2, 100, f"Copying ZIM file ({Path(parent_zim_path).stat().st_size / 1024 / 1024:.0f} MB)...")
                    shutil.copy2(parent_zim_path, dest_zim_path)
                    update_progress(4, 100, "ZIM file copied")

                zim_path_copied = str(dest_zim_path)

        # Load metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        documents = metadata.get("documents", {})
        total_docs = len(documents)

        # Phase 1: Translate HTML files (0-40%)
        if not checkpoint or checkpoint.phase == "html":
            update_progress(5, 100, f"Translating {total_docs} HTML files...")

            start_index = checkpoint.html_files_processed if checkpoint else 0
            doc_ids = list(documents.keys())

            for i, doc_id in enumerate(doc_ids[start_index:], start=start_index):
                if check_cancelled():
                    _save_checkpoint(source_id, target_lang, "html", i, i, 0)
                    result.error = "Cancelled by user"
                    return result

                doc = documents[doc_id]
                local_url = doc.get("local_url", "")

                # Extract filename from local_url
                if local_url:
                    # local_url format: /backup/source_id/filename.html
                    filename = local_url.split("/")[-1] if "/" in local_url else local_url
                    source_html_path = pages_dir / filename

                    if source_html_path.exists():
                        try:
                            with open(source_html_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()

                            # Translate HTML
                            translated_html = translator.translate_html(html_content)

                            # Save to localized folder
                            dest_html_path = localized_pages_dir / filename
                            with open(dest_html_path, 'w', encoding='utf-8') as f:
                                f.write(translated_html)

                            result.html_files_translated += 1
                        except Exception as e:
                            result.warnings.append(f"Failed to translate {filename}: {str(e)}")

                # Update progress (0-40%)
                progress = int(5 + (i / total_docs) * 35)
                update_progress(progress, 100, f"Translated {i+1}/{total_docs} HTML files...")

                # Checkpoint
                if i > 0 and i % checkpoint_interval == 0:
                    _save_checkpoint(source_id, target_lang, "html", i, i, 0)

        # Phase 2: Translate metadata titles (40-50%)
        update_progress(40, 100, "Translating document titles...")

        translated_metadata = {
            "schema_version": metadata.get("schema_version", 3),
            "source_id": result.localized_source_id,
            "source_type": metadata.get("source_type", "html"),
            "last_updated": datetime.utcnow().isoformat(),
            "total_documents": total_docs,
            "document_count": total_docs,
            "total_chars": 0,
            "documents": {}
        }

        # Batch translate titles
        doc_ids = list(documents.keys())
        titles = [documents[doc_id].get("title", "") for doc_id in doc_ids]

        # Translate in batches
        translated_titles = []
        for i in range(0, len(titles), batch_size):
            if check_cancelled():
                result.error = "Cancelled by user"
                return result

            batch = titles[i:i + batch_size]
            translated_batch = translator.translate_batch(batch)
            translated_titles.extend(translated_batch)

            progress = int(40 + (i / len(titles)) * 10)
            update_progress(progress, 100, f"Translated {i}/{len(titles)} titles...")

        # Build translated metadata
        for i, doc_id in enumerate(doc_ids):
            doc = documents[doc_id].copy()
            doc["title"] = translated_titles[i] if i < len(translated_titles) else doc.get("title", "")

            # Update local_url to point to localized folder
            if doc.get("local_url"):
                doc["local_url"] = doc["local_url"].replace(f"/{source_id}/", f"/{result.localized_source_id}/")

            translated_metadata["documents"][doc_id] = doc
            translated_metadata["total_chars"] += doc.get("char_count", 0)

        result.documents_translated = len(translated_titles)

        # Save translated metadata
        translated_metadata_path = localized_path / "_metadata.json"
        with open(translated_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(translated_metadata, f, indent=2, ensure_ascii=False)

        # Phase 3: Generate index from translated content (50-60%)
        update_progress(50, 100, "Generating content index...")

        translated_index = {
            "schema_version": 3,
            "source_id": result.localized_source_id,
            "document_count": total_docs,
            "documents": {}
        }

        # Extract content from translated HTML files
        for i, doc_id in enumerate(doc_ids):
            if check_cancelled():
                result.error = "Cancelled by user"
                return result

            doc = translated_metadata["documents"][doc_id]
            local_url = doc.get("local_url", "")

            if local_url:
                filename = local_url.split("/")[-1] if "/" in local_url else local_url
                html_path = localized_pages_dir / filename

                if html_path.exists():
                    try:
                        content = _extract_text_from_html(html_path)
                        translated_index["documents"][doc_id] = {
                            "title": doc.get("title", ""),
                            "url": doc.get("url", ""),
                            "content": content,
                            "categories": doc.get("categories", []),
                            "doc_type": doc.get("doc_type", "article"),
                            "char_count": len(content)
                        }
                    except Exception as e:
                        result.warnings.append(f"Failed to extract content from {filename}: {str(e)}")

            if i % 100 == 0:
                progress = int(50 + (i / total_docs) * 10)
                update_progress(progress, 100, f"Indexed {i}/{total_docs} documents...")

        # Save index
        translated_index_path = localized_path / "_index.json"
        with open(translated_index_path, 'w', encoding='utf-8') as f:
            json.dump(translated_index, f, indent=2, ensure_ascii=False)

        # Phase 4: Generate embeddings (60-85%)
        # Use local embedding model for offline-ready localizations
        # This ensures localized sources work without internet/API access
        update_progress(60, 100, f"Generating embeddings with {embedding_model}...")

        from offline_tools.embeddings import EmbeddingService
        embeddings_service = EmbeddingService(model=embedding_model)

        if not embeddings_service.is_available():
            result.warnings.append("Embedding service not available - skipping vector generation")
        else:
            # Prepare texts for embedding (title + content preview)
            texts_to_embed = []
            embed_doc_ids = []

            for doc_id, doc in translated_index["documents"].items():
                title = doc.get("title", "")
                content = doc.get("content", "")[:500]  # First 500 chars
                text = f"{title}\n\n{content}" if content else title
                texts_to_embed.append(text)
                embed_doc_ids.append(doc_id)

            # Generate embeddings in batches
            all_embeddings = []
            for i in range(0, len(texts_to_embed), batch_size):
                if check_cancelled():
                    result.error = "Cancelled by user"
                    return result

                batch = texts_to_embed[i:i + batch_size]
                embeddings = embeddings_service.embed_batch(batch, batch_size=batch_size)
                if embeddings:
                    all_embeddings.extend(embeddings)

                progress = int(60 + (i / len(texts_to_embed)) * 25)
                update_progress(progress, 100, f"Generated {i}/{len(texts_to_embed)} embeddings...")

                # Checkpoint
                if i > 0 and i % checkpoint_interval == 0:
                    _save_checkpoint(source_id, target_lang, "vectors", i, total_docs, i)

            result.embeddings_generated = len(all_embeddings)

            # Save vectors file
            vectors_data = {
                "schema_version": 3,
                "source_id": result.localized_source_id,
                "embedding_model": embeddings_service.model_name if hasattr(embeddings_service, 'model_name') else "unknown",
                "dimensions": embeddings_service.get_dimension(),
                "document_count": len(all_embeddings),
                "created_at": datetime.utcnow().isoformat(),
                "vectors": {}
            }

            for i, doc_id in enumerate(embed_doc_ids):
                if i < len(all_embeddings) and all_embeddings[i]:
                    vectors_data["vectors"][doc_id] = all_embeddings[i]

            dimension = embeddings_service.get_dimension()
            vectors_path = localized_path / f"_vectors_{dimension}.json"
            with open(vectors_path, 'w', encoding='utf-8') as f:
                json.dump(vectors_data, f)

        # Phase 5: Populate ChromaDB (85-95%)
        update_progress(85, 100, "Populating language-specific ChromaDB...")

        try:
            from offline_tools.vectordb.store import VectorStore

            # Get language-specific ChromaDB path
            dimension = embeddings_service.get_dimension() if embeddings_service.is_available() else 768
            chroma_path = _get_chroma_path(dimension, target_lang)

            store = VectorStore(persist_dir=str(chroma_path), dimension=dimension)

            # Prepare documents for ChromaDB
            chroma_docs = []
            chroma_embeddings = []

            for i, doc_id in enumerate(embed_doc_ids):
                if i < len(all_embeddings) and all_embeddings[i]:
                    doc = translated_index["documents"].get(doc_id, {})
                    chroma_docs.append({
                        "id": doc_id,
                        "content": doc.get("content", "")[:2000],  # Limit content size
                        "metadata": {
                            "source": result.localized_source_id,
                            "title": doc.get("title", ""),
                            "url": doc.get("url", ""),
                            "doc_type": doc.get("doc_type", "article"),
                            "language": target_lang
                        }
                    })
                    chroma_embeddings.append(all_embeddings[i])

            if chroma_docs:
                store.add_documents(chroma_docs, embeddings=chroma_embeddings)
                update_progress(90, 100, f"Added {len(chroma_docs)} documents to ChromaDB...")

        except Exception as e:
            result.warnings.append(f"Failed to populate ChromaDB: {str(e)}")

        # Phase 6: Create manifest (95-100%)
        update_progress(95, 100, "Creating localized manifest...")

        # Load original manifest if exists
        original_manifest = {}
        if manifest_file.exists():
            with open(manifest_file, 'r', encoding='utf-8') as f:
                original_manifest = json.load(f)

        localized_manifest = {
            "source_id": result.localized_source_id,
            "parent_source": source_id,
            "name": f"{original_manifest.get('name', source_id)} ({target_lang.upper()})",
            "description": original_manifest.get("description", ""),
            "language": target_lang,
            "language_verified": True,  # Known from translation target
            "is_localization": True,
            "localized_from_language": original_manifest.get("language", "en"),
            "localized_at": datetime.utcnow().isoformat(),
            "translation_model": f"Helsinki-NLP/opus-mt-en-{target_lang}",
            "base_url": original_manifest.get("base_url", ""),
            "license": original_manifest.get("license", ""),
            "license_verified": original_manifest.get("license_verified", False),
            "tags": original_manifest.get("tags", []),
            "total_docs": total_docs,
            "total_chars": translated_metadata.get("total_chars", 0),
            "has_backup": True,
            "has_metadata": True,
            "has_index": True,
            "has_vectors": result.embeddings_generated > 0,
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }

        # If parent was a ZIM source, include ZIM info so localized source is self-contained
        if zim_path_copied:
            localized_manifest["created_from"] = "zim_import"
            localized_manifest["zim_path"] = zim_path_copied

        localized_manifest_path = localized_path / "_manifest.json"
        with open(localized_manifest_path, 'w', encoding='utf-8') as f:
            json.dump(localized_manifest, f, indent=2, ensure_ascii=False)

        # Clean up checkpoint
        _delete_checkpoint(source_id, target_lang)

        update_progress(100, 100, "Localization complete!")

        result.success = True
        return result

    except Exception as e:
        result.error = str(e)
        return result


def get_localized_sources(source_id: str) -> List[Dict[str, Any]]:
    """
    Get list of available localizations for a source.

    Args:
        source_id: Parent source ID (e.g., "appropedia")

    Returns:
        List of localization info dicts
    """
    localizations = []

    backup_path = get_backup_path()
    if not backup_path:
        return localizations

    # Look for folders matching pattern: {source_id}_{lang}
    for item in backup_path.iterdir():
        if item.is_dir() and item.name.startswith(f"{source_id}_"):
            # Check if it has a manifest marking it as a localization
            manifest_path = item / "_manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)

                    if manifest.get("is_localization") and manifest.get("parent_source") == source_id:
                        # Get folder size
                        size_bytes = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                        size_mb = size_bytes / (1024 * 1024)

                        localizations.append({
                            "localized_source_id": item.name,
                            "language": manifest.get("language", "unknown"),
                            "language_name": _get_language_name(manifest.get("language", "")),
                            "document_count": manifest.get("total_docs", 0),
                            "created_at": manifest.get("localized_at", manifest.get("created_at", "")),
                            "size_mb": round(size_mb, 1),
                            "has_vectors": manifest.get("has_vectors", False)
                        })
                except Exception:
                    pass

    return localizations


def delete_localized_source(source_id: str, target_lang: str) -> Dict[str, Any]:
    """
    Delete a localized source and all its files.

    Args:
        source_id: Parent source ID
        target_lang: Language code of localization to delete

    Returns:
        {"success": True, "deleted_files": [...]} or {"success": False, "error": "..."}
    """
    localized_source_id = f"{source_id}_{target_lang}"

    backup_path = get_backup_path()
    if not backup_path:
        return {"success": False, "error": "No backup folder configured"}

    localized_path = backup_path / localized_source_id

    if not localized_path.exists():
        return {"success": False, "error": f"Localized source not found: {localized_source_id}"}

    # Verify it's actually a localization (safety check)
    manifest_path = localized_path / "_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            if not manifest.get("is_localization"):
                return {"success": False, "error": "Target is not a localization"}
        except Exception:
            pass

    try:
        # Delete the folder
        deleted_files = []
        for f in localized_path.rglob('*'):
            if f.is_file():
                deleted_files.append(str(f.relative_to(localized_path)))

        shutil.rmtree(localized_path)

        # Note: We skip ChromaDB cleanup for localized sources because:
        # 1. It's slow and blocks the UI
        # 2. Orphaned entries don't hurt - they won't match any searches
        # 3. The language-specific ChromaDB can be deleted manually if needed

        return {
            "success": True,
            "deleted_source": localized_source_id,
            "deleted_files": deleted_files
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_source(source_id: str) -> Dict[str, Any]:
    """
    Delete a source entirely (for cleanup after localization).

    Args:
        source_id: Source ID to delete

    Returns:
        {"success": True, "deleted_files": [...]} or {"success": False, "error": "..."}
    """
    backup_path = get_backup_path()
    if not backup_path:
        return {"success": False, "error": "No backup folder configured"}

    source_path = backup_path / source_id

    if not source_path.exists():
        return {"success": False, "error": f"Source not found: {source_id}"}

    try:
        # Count files for reporting
        deleted_files = []
        for f in source_path.rglob('*'):
            if f.is_file():
                deleted_files.append(str(f.relative_to(source_path)))

        shutil.rmtree(source_path)

        return {
            "success": True,
            "deleted_source": source_id,
            "deleted_files": deleted_files,
            "file_count": len(deleted_files)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def validate_localization_requirements(source_id: str, target_lang: str, force_overwrite: bool = False) -> Dict[str, Any]:
    """
    Check if a source can be localized to the target language.

    Args:
        source_id: Source to localize
        target_lang: Target language code
        force_overwrite: If True, skip "already exists" check

    Returns:
        {
            "can_localize": True/False,
            "has_source": True/False,
            "has_metadata": True/False,
            "has_pages": True/False,
            "has_translation_model": True/False,
            "parent_language": "en",
            "issues": []
        }
    """
    result = {
        "can_localize": False,
        "has_source": False,
        "has_metadata": False,
        "has_pages": False,
        "has_translation_model": False,
        "parent_language": "en",
        "issues": []
    }

    backup_path = get_backup_path()
    if not backup_path:
        result["issues"].append("No backup folder configured")
        return result

    source_path = backup_path / source_id

    # Check source exists
    if source_path.exists():
        result["has_source"] = True
    else:
        result["issues"].append(f"Source not found: {source_id}")
        return result

    # Check metadata
    metadata_path = source_path / "_metadata.json"
    if metadata_path.exists():
        result["has_metadata"] = True
    else:
        result["issues"].append("Missing _metadata.json")

    # Check pages folder
    pages_path = source_path / "pages"
    if pages_path.exists() and any(pages_path.iterdir()):
        result["has_pages"] = True
    else:
        result["issues"].append("Missing or empty pages/ folder")

    # Check translation model
    try:
        from offline_tools.translation import TranslationService
        translator = TranslationService(target_lang)
        if translator.is_available():
            result["has_translation_model"] = True
        else:
            result["issues"].append(f"Translation model not installed for: {target_lang}")
    except Exception as e:
        result["issues"].append(f"Translation service error: {str(e)}")

    # Get parent language from manifest and check for localization issues
    manifest_path = source_path / "_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            result["parent_language"] = manifest.get("language", "en")

            # Check if source is already a localization
            if manifest.get("is_localization"):
                parent_source = manifest.get("parent_source", "unknown")
                result["issues"].append(
                    f"Source is already a localization - localize the parent source '{parent_source}' instead"
                )
                result["is_already_localization"] = True

            # Check if source language matches target language
            source_lang = manifest.get("language", "").lower()
            if source_lang and source_lang == target_lang.lower():
                lang_name = _get_language_name(target_lang)
                result["issues"].append(
                    f"Source is already in {lang_name} - choose a different target language"
                )
                result["same_language"] = True

        except Exception:
            pass

    # Check if already localized (only block if complete - has manifest)
    # Skip this check if force_overwrite is True
    if not force_overwrite:
        localized_path = backup_path / f"{source_id}_{target_lang}"
        localized_manifest = localized_path / "_manifest.json"
        if localized_manifest.exists():
            # Check if it's a complete localization
            try:
                with open(localized_manifest, 'r', encoding='utf-8') as f:
                    loc_manifest = json.load(f)
                if loc_manifest.get("is_localization"):
                    result["issues"].append(f"Localization already exists: {source_id}_{target_lang}")
                    result["existing_localization"] = True
            except Exception:
                pass

    # Check for resumable checkpoint
    checkpoint = _load_checkpoint(source_id, target_lang)
    if checkpoint:
        result["has_checkpoint"] = True
        result["checkpoint_progress"] = checkpoint.progress
        result["checkpoint_phase"] = checkpoint.phase

    # Determine if can localize
    result["can_localize"] = (
        result["has_source"] and
        result["has_metadata"] and
        result["has_pages"] and
        result["has_translation_model"] and
        len(result["issues"]) == 0
    )

    return result


# Helper functions

def _get_chroma_path(dimension: int = 768, language: str = "en") -> Path:
    """Get ChromaDB path for a specific dimension and language"""
    backup_path = get_backup_path()
    if not backup_path:
        raise ValueError("No backup folder configured")

    if language == "en":
        return backup_path / f"chroma_db_{dimension}"
    return backup_path / f"chroma_db_{dimension}_{language}"


def _get_language_name(lang_code: str) -> str:
    """Get human-readable language name from code"""
    language_names = {
        "es": "Spanish",
        "fr": "French",
        "ar": "Arabic",
        "zh": "Chinese",
        "pt": "Portuguese",
        "hi": "Hindi",
        "sw": "Swahili",
        "ht": "Haitian Creole",
        "de": "German",
        "it": "Italian",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean"
    }
    return language_names.get(lang_code, lang_code.upper())


def _extract_text_from_html(html_path: Path) -> str:
    """Extract plain text from HTML file"""
    try:
        from bs4 import BeautifulSoup

        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()

        # Get text
        text = soup.get_text(separator=' ', strip=True)
        return text

    except Exception:
        return ""


def _save_checkpoint(source_id: str, target_lang: str, phase: str,
                     doc_index: int, total_docs: int, embeddings_done: int):
    """Save checkpoint for resume capability"""
    backup_path = get_backup_path()
    if not backup_path:
        return

    checkpoint_dir = backup_path / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "source_id": source_id,
        "target_lang": target_lang,
        "phase": phase,
        "last_doc_index": doc_index,
        "html_files_processed": doc_index if phase == "html" else total_docs,
        "documents_processed": doc_index if phase in ["metadata", "index"] else 0,
        "embeddings_processed": embeddings_done,
        "progress": int((doc_index / total_docs) * 100) if total_docs > 0 else 0,
        "created_at": datetime.utcnow().isoformat(),
        "last_saved": datetime.utcnow().isoformat()
    }

    checkpoint_path = checkpoint_dir / f"localize_{source_id}_{target_lang}.json"
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2)


def _load_checkpoint(source_id: str, target_lang: str) -> Optional[LocalizationCheckpoint]:
    """Load checkpoint if exists"""
    backup_path = get_backup_path()
    if not backup_path:
        return None

    checkpoint_path = backup_path / "_checkpoints" / f"localize_{source_id}_{target_lang}.json"

    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return LocalizationCheckpoint(
            source_id=data.get("source_id", source_id),
            target_lang=data.get("target_lang", target_lang),
            progress=data.get("progress", 0),
            last_doc_index=data.get("last_doc_index", 0),
            documents_processed=data.get("documents_processed", 0),
            html_files_processed=data.get("html_files_processed", 0),
            phase=data.get("phase", "html"),
            created_at=data.get("created_at", ""),
            last_saved=data.get("last_saved", "")
        )
    except Exception:
        return None


def _delete_checkpoint(source_id: str, target_lang: str):
    """Delete checkpoint after successful completion"""
    backup_path = get_backup_path()
    if not backup_path:
        return

    checkpoint_path = backup_path / "_checkpoints" / f"localize_{source_id}_{target_lang}.json"

    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except Exception:
            pass
