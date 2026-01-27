"""
Model Management API

Endpoints for downloading and managing embedding and LLM models
for offline semantic search and AI conversation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import json
import os
import httpx

router = APIRouter(prefix="/api", tags=["Model Management"])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_local_config():
    """Get local config - imported here to avoid circular imports"""
    from admin.local_config import get_local_config as _get_config
    return _get_config()


def get_model_registry():
    """Get model registry - imported here to avoid circular imports"""
    from offline_tools.model_registry import get_model_registry as _get_registry
    return _get_registry()


# =============================================================================
# REQUEST MODELS
# =============================================================================

class DownloadModelRequest(BaseModel):
    model_id: str


class SetActiveModelRequest(BaseModel):
    model_id: str


# =============================================================================
# MODEL LISTING
# =============================================================================

@router.get("/available-models")
async def get_available_models(model_type: Optional[str] = None):
    """
    Get list of available models with their install status.

    Args:
        model_type: Filter by type ("embedding" or "llm"), or None for all

    Returns:
        List of models with metadata and install status
    """
    try:
        registry = get_model_registry()
        models = registry.get_available_models(model_type)
        system_info = registry.get_system_info()

        return {
            "models": list(models.values()),
            "system_info": system_info,
            "installed_embedding": registry.get_installed_embedding_model(),
            "installed_llm": registry.get_installed_llm_model()
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get models: {str(e)}")


@router.get("/model-status")
async def get_model_status():
    """
    Get current status of installed models and embedding service.
    """
    try:
        registry = get_model_registry()

        # Get embedding service status
        embedding_status = None
        try:
            from offline_tools.embeddings import EmbeddingService
            service = EmbeddingService()
            embedding_status = service.get_status()
        except Exception as e:
            embedding_status = {"available": False, "error": str(e)}

        return {
            "installed_embedding": registry.get_installed_embedding_model(),
            "installed_llm": registry.get_installed_llm_model(),
            "embedding_service": embedding_status,
            "system_info": registry.get_system_info()
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get status: {str(e)}")


@router.get("/model-info/{model_id}")
async def get_model_info(model_id: str):
    """Get detailed information about a specific model"""
    try:
        registry = get_model_registry()
        info = registry.get_model_info(model_id)

        if not info:
            raise HTTPException(404, f"Model not found: {model_id}")

        return info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to get model info: {str(e)}")


# =============================================================================
# MODEL DOWNLOAD
# =============================================================================

def _download_embedding_model(model_id: str, progress_callback=None, cancel_checker=None, job_id=None):
    """
    Background job to download an embedding model from HuggingFace.
    """
    from offline_tools.model_registry import AVAILABLE_MODELS, get_model_registry

    def update_progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    if model_id not in AVAILABLE_MODELS:
        raise Exception(f"Unknown model: {model_id}")

    model_info = AVAILABLE_MODELS[model_id]
    if model_info["type"] != "embedding":
        raise Exception(f"Not an embedding model: {model_id}")

    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise Exception("No backup folder configured. Set it in Settings first.")

    # Create model directory
    model_dir = Path(backup_folder) / "models" / "embeddings" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    update_progress(5, f"Downloading {model_info['display_name']}...")

    # Download files from HuggingFace
    repo = model_info["huggingface_repo"]
    files_to_download = model_info.get("files", [])

    if not files_to_download:
        # Default files for sentence-transformers models
        files_to_download = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "modules.json",
            "sentence_bert_config.json",
            "special_tokens_map.json",
            "pytorch_model.bin"  # Main weights file
        ]

    downloaded = []
    failed = []
    total_files = len(files_to_download)

    for idx, filename in enumerate(files_to_download):
        if cancel_checker and cancel_checker():
            raise Exception("Download cancelled")

        pct = 5 + int((idx / total_files) * 85)
        update_progress(pct, f"Downloading {filename}...")

        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        local_path = model_dir / filename

        # Skip if file already exists and has content
        if local_path.exists() and local_path.stat().st_size > 0:
            downloaded.append(filename)
            continue

        try:
            # Download with streaming for large files
            with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
                if response.status_code == 404:
                    # Some files are optional
                    if filename not in ["pytorch_model.bin", "model.safetensors", "config.json"]:
                        print(f"Optional file not found: {filename}")
                        continue
                    failed.append(filename)
                    continue

                response.raise_for_status()

                # Create any intermediate subdirectories if needed
                local_path.parent.mkdir(parents=True, exist_ok=True)

                with open(local_path, 'wb') as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)

            downloaded.append(filename)

        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            if filename in ["pytorch_model.bin", "model.safetensors"]:
                failed.append(filename)

    # Check if we got the essential files
    has_weights = (model_dir / "pytorch_model.bin").exists() or \
                  (model_dir / "model.safetensors").exists()

    if not has_weights:
        # Try model.safetensors as alternative
        update_progress(90, "Trying alternative weights file...")
        url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"
        local_path = model_dir / "model.safetensors"

        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
                response.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
            downloaded.append("model.safetensors")
            has_weights = True
        except Exception as e:
            print(f"Failed to download model.safetensors: {e}")

    if not has_weights:
        raise Exception("Failed to download model weights (pytorch_model.bin or model.safetensors)")

    # Update registry
    update_progress(95, "Updating model registry...")
    registry = get_model_registry()
    registry.set_active_embedding_model(model_id)

    update_progress(100, "Download complete")

    return {
        "status": "success",
        "model_id": model_id,
        "files_downloaded": len(downloaded),
        "files_failed": len(failed),
        "path": str(model_dir),
        "message": f"Downloaded {model_info['display_name']} successfully"
    }


def _download_llm_model(model_id: str, progress_callback=None, cancel_checker=None, job_id=None):
    """
    Background job to download an LLM model (GGUF) from HuggingFace.
    """
    from offline_tools.model_registry import AVAILABLE_MODELS, get_model_registry

    def update_progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    if model_id not in AVAILABLE_MODELS:
        raise Exception(f"Unknown model: {model_id}")

    model_info = AVAILABLE_MODELS[model_id]
    if model_info["type"] != "llm":
        raise Exception(f"Not an LLM model: {model_id}")

    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise Exception("No backup folder configured. Set it in Settings first.")

    # Create model directory
    model_dir = Path(backup_folder) / "models" / "llm" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    filename = model_info.get("filename", "")
    if not filename:
        raise Exception("No filename specified for LLM model")

    local_path = model_dir / filename

    # Check if already downloaded
    if local_path.exists():
        expected_size = model_info.get("size_mb", 0) * 1024 * 1024
        actual_size = local_path.stat().st_size
        # Allow 5% variance in size
        if actual_size > expected_size * 0.95:
            update_progress(100, "Model already downloaded")
            registry = get_model_registry()
            registry.set_active_llm_model(model_id)
            return {
                "status": "success",
                "model_id": model_id,
                "already_downloaded": True,
                "path": str(local_path),
                "message": f"{model_info['display_name']} already downloaded"
            }

    update_progress(5, f"Downloading {model_info['display_name']} ({model_info.get('size_mb', 0)}MB)...")

    repo = model_info["huggingface_repo"]
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"

    try:
        # Download with progress tracking
        with httpx.stream("GET", url, follow_redirects=True, timeout=3600.0) as response:
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(local_path, 'wb') as f:
                for chunk in response.iter_bytes(chunk_size=65536):
                    if cancel_checker and cancel_checker():
                        f.close()
                        local_path.unlink(missing_ok=True)
                        raise Exception("Download cancelled")

                    f.write(chunk)
                    downloaded_size += len(chunk)

                    if total_size > 0:
                        pct = 5 + int((downloaded_size / total_size) * 90)
                        downloaded_mb = downloaded_size / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        update_progress(pct, f"Downloading: {downloaded_mb:.1f}MB / {total_mb:.1f}MB")

    except httpx.HTTPError as e:
        local_path.unlink(missing_ok=True)
        raise Exception(f"Download failed: {str(e)}")

    # Update registry
    update_progress(98, "Updating model registry...")
    registry = get_model_registry()
    registry.set_active_llm_model(model_id)

    update_progress(100, "Download complete")

    return {
        "status": "success",
        "model_id": model_id,
        "path": str(local_path),
        "size_mb": local_path.stat().st_size / (1024 * 1024),
        "message": f"Downloaded {model_info['display_name']} successfully"
    }


@router.post("/download-model")
async def download_model(request: DownloadModelRequest):
    """
    Download a model from HuggingFace.
    Submits as background job - returns job_id for tracking.
    """
    from offline_tools.model_registry import AVAILABLE_MODELS

    model_id = request.model_id

    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(404, f"Unknown model: {model_id}")

    model_info = AVAILABLE_MODELS[model_id]

    try:
        config = get_local_config()
        backup_folder = config.get_backup_folder()

        if not backup_folder:
            raise HTTPException(400, "No backup folder configured. Set it in Settings first.")

        from admin.job_manager import get_job_manager
        manager = get_job_manager()

        # Choose download function based on model type
        if model_info["type"] == "embedding":
            download_func = _download_embedding_model
        else:
            download_func = _download_llm_model

        try:
            job_id = manager.submit(
                "model_download",
                model_id,
                download_func,
                model_id
            )
        except ValueError as e:
            raise HTTPException(409, str(e))

        return {
            "status": "submitted",
            "job_id": job_id,
            "model_id": model_id,
            "model_type": model_info["type"],
            "message": f"Download job started for {model_info['display_name']}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to start download: {str(e)}")


# =============================================================================
# SET ACTIVE MODEL
# =============================================================================

@router.post("/set-active-model")
async def set_active_model(request: SetActiveModelRequest):
    """
    Set which model is active for use.
    """
    from offline_tools.model_registry import AVAILABLE_MODELS

    model_id = request.model_id

    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(404, f"Unknown model: {model_id}")

    try:
        registry = get_model_registry()

        # Check if model is installed
        if not registry.is_model_installed(model_id):
            raise HTTPException(400, f"Model not installed: {model_id}. Download it first.")

        model_info = AVAILABLE_MODELS[model_id]

        if model_info["type"] == "embedding":
            success = registry.set_active_embedding_model(model_id)
        else:
            success = registry.set_active_llm_model(model_id)

        if not success:
            raise HTTPException(500, "Failed to update registry")

        return {
            "status": "success",
            "model_id": model_id,
            "model_type": model_info["type"],
            "message": f"Set {model_info['display_name']} as active"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to set active model: {str(e)}")


# =============================================================================
# LANGUAGE PACK MANAGEMENT
# =============================================================================

def get_language_registry():
    """Get language registry - imported here to avoid circular imports"""
    from offline_tools.language_registry import get_language_registry as _get_registry
    return _get_registry()


class DownloadLanguageRequest(BaseModel):
    language_code: str


class SetLanguageRequest(BaseModel):
    language_code: str


class TranslateArticleRequest(BaseModel):
    article_id: str
    html: str
    language: Optional[str] = None


@router.get("/available-languages")
async def get_available_languages():
    """
    Get list of available language packs with install status.
    """
    try:
        registry = get_language_registry()
        packs = registry.get_available_packs()
        config = get_local_config()

        return {
            "languages": list(packs.values()),
            "active_language": registry.get_active_language(),
            "translation_enabled": config.is_translation_enabled(),
            "installed_count": len(registry.get_installed_packs())
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get languages: {str(e)}")


@router.get("/language-status")
async def get_language_status():
    """
    Get current language/translation settings and cache status.
    """
    try:
        registry = get_language_registry()
        config = get_local_config()

        # Get cache stats
        cache_stats = {"enabled": False, "size_mb": 0, "article_count": 0}
        try:
            from offline_tools.translation import TranslationCache
            cache = TranslationCache()
            cache_stats = cache.get_stats()
        except Exception:
            pass

        return {
            "active_language": registry.get_active_language(),
            "translation_enabled": config.is_translation_enabled(),
            "installed_packs": registry.get_installed_packs(),
            "cache": cache_stats
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get status: {str(e)}")


@router.get("/language-info/{lang_code}")
async def get_language_info(lang_code: str):
    """Get detailed information about a specific language pack"""
    try:
        registry = get_language_registry()
        info = registry.get_pack_info(lang_code)

        if not info:
            raise HTTPException(404, f"Language not found: {lang_code}")

        return info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to get language info: {str(e)}")


def _download_language_pack(lang_code: str, progress_callback=None, cancel_checker=None, job_id=None):
    """
    Background job to download a language pack from HuggingFace.
    Downloads both en->lang and lang->en models.
    """
    from offline_tools.language_registry import AVAILABLE_LANGUAGE_PACKS, get_language_registry

    def update_progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    if lang_code not in AVAILABLE_LANGUAGE_PACKS:
        raise Exception(f"Unknown language: {lang_code}")

    pack_info = AVAILABLE_LANGUAGE_PACKS[lang_code]

    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise Exception("No backup folder configured. Set it in Settings first.")

    # Create language pack directory
    pack_dir = Path(backup_folder) / "models" / "translation" / "marian" / lang_code
    pack_dir.mkdir(parents=True, exist_ok=True)

    repos = pack_info.get("huggingface_repos", {})
    total_models = len(repos)
    models_downloaded = 0

    for direction, repo in repos.items():
        if cancel_checker and cancel_checker():
            raise Exception("Download cancelled")

        model_name = repo.split("/")[-1]
        model_dir = pack_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        base_pct = int((models_downloaded / total_models) * 90)
        update_progress(base_pct + 5, f"Downloading {model_name}...")

        # Files needed for MarianMT
        files_to_download = [
            "config.json",
            "tokenizer_config.json",
            "source.spm",
            "target.spm",
            "vocab.json",
            "pytorch_model.bin"
        ]

        for idx, filename in enumerate(files_to_download):
            if cancel_checker and cancel_checker():
                raise Exception("Download cancelled")

            file_pct = base_pct + 5 + int((idx / len(files_to_download)) * (90 / total_models))
            update_progress(file_pct, f"Downloading {model_name}/{filename}...")

            url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
            local_path = model_dir / filename

            # Skip if exists
            if local_path.exists() and local_path.stat().st_size > 0:
                continue

            try:
                with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
                    if response.status_code == 404:
                        # Some files are optional
                        if filename not in ["pytorch_model.bin", "config.json"]:
                            continue
                        raise Exception(f"Required file not found: {filename}")

                    response.raise_for_status()

                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_bytes(chunk_size=8192):
                            f.write(chunk)

            except httpx.HTTPError as e:
                if filename in ["pytorch_model.bin", "config.json"]:
                    raise Exception(f"Failed to download {filename}: {str(e)}")

        models_downloaded += 1

    # Update registry
    update_progress(95, "Updating language registry...")
    registry = get_language_registry()
    registry.mark_pack_installed(lang_code)

    update_progress(100, "Download complete")

    return {
        "status": "success",
        "language_code": lang_code,
        "path": str(pack_dir),
        "message": f"Downloaded {pack_info['display_name']} language pack"
    }


@router.post("/download-language")
async def download_language(request: DownloadLanguageRequest):
    """
    Download a language pack from HuggingFace.
    Submits as background job - returns job_id for tracking.
    """
    from offline_tools.language_registry import AVAILABLE_LANGUAGE_PACKS

    lang_code = request.language_code

    if lang_code not in AVAILABLE_LANGUAGE_PACKS:
        raise HTTPException(404, f"Unknown language: {lang_code}")

    pack_info = AVAILABLE_LANGUAGE_PACKS[lang_code]

    try:
        config = get_local_config()
        backup_folder = config.get_backup_folder()

        if not backup_folder:
            raise HTTPException(400, "No backup folder configured. Set it in Settings first.")

        from admin.job_manager import get_job_manager
        manager = get_job_manager()

        try:
            job_id = manager.submit(
                "language_download",
                lang_code,
                _download_language_pack,
                lang_code
            )
        except ValueError as e:
            raise HTTPException(409, str(e))

        return {
            "status": "submitted",
            "job_id": job_id,
            "language_code": lang_code,
            "message": f"Download job started for {pack_info['display_name']}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to start download: {str(e)}")


@router.post("/set-language")
async def set_language(request: SetLanguageRequest):
    """
    Set the active language for translation.
    Use "en" to disable translation.
    """
    from offline_tools.language_registry import AVAILABLE_LANGUAGE_PACKS

    lang_code = request.language_code

    # "en" means no translation
    if lang_code != "en" and lang_code not in AVAILABLE_LANGUAGE_PACKS:
        raise HTTPException(404, f"Unknown language: {lang_code}")

    try:
        registry = get_language_registry()

        # Check if pack is installed (unless setting to English)
        if lang_code != "en" and not registry.is_pack_installed(lang_code):
            raise HTTPException(400, f"Language pack not installed: {lang_code}. Download it first.")

        success = registry.set_active_language(lang_code)

        if not success:
            raise HTTPException(500, "Failed to update registry")

        # Update config
        config = get_local_config()
        config.set_translation_language(lang_code)
        config.save()

        pack_name = "English (no translation)" if lang_code == "en" else \
                    AVAILABLE_LANGUAGE_PACKS[lang_code]["display_name"]

        return {
            "status": "success",
            "language_code": lang_code,
            "translation_enabled": lang_code != "en",
            "message": f"Set language to {pack_name}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to set language: {str(e)}")


@router.post("/translate-article")
async def translate_article(request: TranslateArticleRequest):
    """
    Translate article HTML content.
    Uses cache if available.
    """
    try:
        from offline_tools.translation import get_translation_service

        service = get_translation_service(request.language)

        if not service.is_available():
            raise HTTPException(400, f"Translation not available for language: {service.language}")

        translated = service.translate_html(request.html, request.article_id)

        return {
            "status": "success",
            "article_id": request.article_id,
            "language": service.language,
            "html": translated
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Translation failed: {str(e)}")


@router.get("/translation-cache")
async def get_translation_cache():
    """Get translation cache statistics"""
    try:
        from offline_tools.translation import TranslationCache
        cache = TranslationCache()
        return cache.get_stats()
    except Exception as e:
        raise HTTPException(500, f"Failed to get cache stats: {str(e)}")


@router.post("/clear-translation-cache")
async def clear_translation_cache(language: Optional[str] = None):
    """
    Clear translation cache.
    If language specified, clears only that language's cache.
    """
    try:
        from offline_tools.translation import TranslationCache
        cache = TranslationCache()

        if language:
            count = cache.clear_language(language)
            return {
                "status": "success",
                "language": language,
                "cleared_count": count
            }
        else:
            # Clear all languages
            total = 0
            registry = get_language_registry()
            for lang in registry.get_installed_packs():
                total += cache.clear_language(lang)

            return {
                "status": "success",
                "cleared_count": total
            }

    except Exception as e:
        raise HTTPException(500, f"Failed to clear cache: {str(e)}")


class RemoveLanguageRequest(BaseModel):
    language_code: str


@router.post("/remove-language")
async def remove_language(request: RemoveLanguageRequest):
    """
    Remove an installed language pack.
    Deletes downloaded model files.
    """
    import shutil
    from offline_tools.language_registry import AVAILABLE_LANGUAGE_PACKS

    lang_code = request.language_code

    if lang_code not in AVAILABLE_LANGUAGE_PACKS:
        raise HTTPException(404, f"Unknown language: {lang_code}")

    try:
        registry = get_language_registry()

        if not registry.is_pack_installed(lang_code):
            raise HTTPException(400, f"Language pack not installed: {lang_code}")

        # Get pack path
        pack_path = registry.get_pack_path(lang_code)

        if pack_path and pack_path.exists():
            # Delete the language pack folder
            shutil.rmtree(pack_path)

        # Update registry
        registry.mark_pack_removed(lang_code)

        # If this was the active language, reset to English
        if registry.get_active_language() == lang_code:
            registry.set_active_language("en")
            config = get_local_config()
            config.set_translation_language("en")
            config.save()

        pack_name = AVAILABLE_LANGUAGE_PACKS[lang_code]["display_name"]

        return {
            "status": "success",
            "language_code": lang_code,
            "message": f"Removed {pack_name} language pack"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to remove language: {str(e)}")
