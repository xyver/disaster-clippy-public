"""
Model Registry for Offline Capabilities
Manages embedding models and LLM models for offline semantic search and conversation.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


# Available models with their metadata
AVAILABLE_MODELS = {
    # Embedding models - ordered by dimension (smallest to largest)
    "all-MiniLM-L6-v2": {
        "type": "embedding",
        "display_name": "MiniLM L6 (384-dim)",
        "description": "Fast and lightweight - ideal for Raspberry Pi and low-end hardware",
        "dimensions": 384,
        "size_mb": 80,
        "min_ram_gb": 2,
        "huggingface_repo": "sentence-transformers/all-MiniLM-L6-v2",
        "files": [
            "1_Pooling/config.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "modules.json",
            "pytorch_model.bin",
            "sentence_bert_config.json",
            "special_tokens_map.json"
        ]
    },
    "all-mpnet-base-v2": {
        "type": "embedding",
        "display_name": "MPNet Base v2 (768-dim)",
        "description": "Recommended - best balance of quality and speed for offline search",
        "dimensions": 768,
        "size_mb": 420,
        "min_ram_gb": 4,
        "huggingface_repo": "sentence-transformers/all-mpnet-base-v2",
        "files": [
            "1_Pooling/config.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "modules.json",
            "pytorch_model.bin",
            "sentence_bert_config.json",
            "special_tokens_map.json"
        ]
    },
    "intfloat-e5-large-v2": {
        "type": "embedding",
        "display_name": "E5 Large v2 (1024-dim)",
        "description": "High quality - better semantic matching, requires 8GB+ RAM",
        "dimensions": 1024,
        "size_mb": 1340,
        "min_ram_gb": 8,
        "huggingface_repo": "intfloat/e5-large-v2",
        "files": [
            "1_Pooling/config.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "modules.json",
            "pytorch_model.bin",
            "sentence_bert_config.json",
            "special_tokens_map.json"
        ]
    },
    # LLM models (GGUF format for llama.cpp)
    "tinyllama-1.1b-q4": {
        "type": "llm",
        "display_name": "TinyLlama 1.1B (Q4)",
        "description": "Basic quality, works on constrained hardware (RPi4 4GB)",
        "parameters": "1.1B",
        "quantization": "Q4_K_M",
        "size_mb": 700,
        "min_ram_gb": 2,
        "tokens_per_sec": "15-30",
        "huggingface_repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    },
    "llama-3.2-3b-q4": {
        "type": "llm",
        "display_name": "Llama 3.2 3B (Q4)",
        "description": "Recommended - good balance of quality and speed",
        "parameters": "3B",
        "quantization": "Q4_K_M",
        "size_mb": 2000,
        "min_ram_gb": 4,
        "tokens_per_sec": "8-15",
        "huggingface_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    },
    "mistral-7b-q4": {
        "type": "llm",
        "display_name": "Mistral 7B (Q4)",
        "description": "Best quality, requires desktop with 8GB+ RAM",
        "parameters": "7B",
        "quantization": "Q4_K_M",
        "size_mb": 4000,
        "min_ram_gb": 8,
        "tokens_per_sec": "5-10",
        "huggingface_repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    }
}


class ModelRegistry:
    """Manages installed models and their configuration"""

    DEFAULT_REGISTRY = {
        "schema_version": 1,
        "last_updated": None,
        "installed_embedding": None,
        "installed_llm": None,
        "models": {}
    }

    def __init__(self, backup_folder: Optional[str] = None):
        """
        Initialize model registry.

        Args:
            backup_folder: Path to backup folder. If None, will try to get from local_config.
        """
        self.backup_folder = backup_folder
        if not self.backup_folder:
            try:
                from admin.local_config import get_local_config
                self.backup_folder = get_local_config().get_backup_folder()
            except Exception:
                pass

        self.registry = self._load_registry()

    def _get_models_folder(self) -> Optional[Path]:
        """Get the models folder path"""
        if not self.backup_folder:
            return None
        return Path(self.backup_folder) / "models"

    def _get_registry_path(self) -> Optional[Path]:
        """Get path to _models.json registry file"""
        models_folder = self._get_models_folder()
        if not models_folder:
            return None
        return models_folder / "_models.json"

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file or return default"""
        registry_path = self._get_registry_path()
        if registry_path and registry_path.exists():
            try:
                with open(registry_path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                # Merge with defaults
                result = self.DEFAULT_REGISTRY.copy()
                result.update(saved)
                return result
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading model registry: {e}")
        return self.DEFAULT_REGISTRY.copy()

    def save(self) -> bool:
        """Save registry to file"""
        registry_path = self._get_registry_path()
        if not registry_path:
            print("Cannot save registry: backup folder not configured")
            return False

        try:
            # Ensure models folder exists
            registry_path.parent.mkdir(parents=True, exist_ok=True)

            self.registry["last_updated"] = datetime.now().isoformat()
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2)
            return True
        except IOError as e:
            print(f"Error saving model registry: {e}")
            return False

    def get_models_path(self, model_type: str = "embeddings") -> Optional[Path]:
        """
        Get path to models folder for a type.

        Args:
            model_type: "embeddings" or "llm"

        Returns:
            Path to folder or None if not configured
        """
        models_folder = self._get_models_folder()
        if not models_folder:
            return None
        return models_folder / model_type

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """
        Get path to a specific model's folder.

        Args:
            model_id: Model identifier (e.g., "all-mpnet-base-v2")

        Returns:
            Path to model folder or None
        """
        if model_id not in AVAILABLE_MODELS:
            return None

        model_info = AVAILABLE_MODELS[model_id]
        model_type = "embeddings" if model_info["type"] == "embedding" else "llm"
        models_path = self.get_models_path(model_type)

        if not models_path:
            return None

        return models_path / model_id

    def is_model_installed(self, model_id: str) -> bool:
        """Check if a model is installed locally"""
        model_path = self.get_model_path(model_id)
        if not model_path or not model_path.exists():
            return False

        model_info = AVAILABLE_MODELS.get(model_id)
        if not model_info:
            return False

        # Check for required files
        if model_info["type"] == "embedding":
            # For embedding models, check for pytorch_model.bin or model.safetensors
            return (model_path / "pytorch_model.bin").exists() or \
                   (model_path / "model.safetensors").exists()
        else:
            # For LLM models, check for the GGUF file
            filename = model_info.get("filename", "")
            return (model_path / filename).exists()

    def get_installed_embedding_model(self) -> Optional[str]:
        """Get the currently installed embedding model ID"""
        # Check registry first
        installed = self.registry.get("installed_embedding")
        if installed and self.is_model_installed(installed):
            return installed

        # Scan for installed embedding models
        for model_id, info in AVAILABLE_MODELS.items():
            if info["type"] == "embedding" and self.is_model_installed(model_id):
                # Update registry
                self.registry["installed_embedding"] = model_id
                self.save()
                return model_id

        return None

    def get_installed_llm_model(self) -> Optional[str]:
        """Get the currently installed LLM model ID"""
        installed = self.registry.get("installed_llm")
        if installed and self.is_model_installed(installed):
            return installed

        # Scan for installed LLM models
        for model_id, info in AVAILABLE_MODELS.items():
            if info["type"] == "llm" and self.is_model_installed(model_id):
                self.registry["installed_llm"] = model_id
                self.save()
                return model_id

        return None

    def set_active_embedding_model(self, model_id: str) -> bool:
        """Set the active embedding model"""
        if model_id not in AVAILABLE_MODELS:
            return False
        if AVAILABLE_MODELS[model_id]["type"] != "embedding":
            return False

        self.registry["installed_embedding"] = model_id
        return self.save()

    def set_active_llm_model(self, model_id: str) -> bool:
        """Set the active LLM model"""
        if model_id not in AVAILABLE_MODELS:
            return False
        if AVAILABLE_MODELS[model_id]["type"] != "llm":
            return False

        self.registry["installed_llm"] = model_id
        return self.save()

    def get_available_models(self, model_type: Optional[str] = None) -> Dict[str, Dict]:
        """
        Get all available models with their install status.

        Args:
            model_type: Filter by type ("embedding" or "llm"), or None for all

        Returns:
            Dict of model_id -> model info with "installed" field added
        """
        result = {}
        for model_id, info in AVAILABLE_MODELS.items():
            if model_type and info["type"] != model_type:
                continue

            model_data = info.copy()
            model_data["id"] = model_id
            model_data["installed"] = self.is_model_installed(model_id)

            # Add active status
            if info["type"] == "embedding":
                model_data["active"] = (model_id == self.registry.get("installed_embedding"))
            else:
                model_data["active"] = (model_id == self.registry.get("installed_llm"))

            result[model_id] = model_data

        return result

    def get_huggingface_download_url(self, model_id: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Get HuggingFace download URL for a model file.

        Args:
            model_id: Model identifier
            filename: Specific file to download (for embedding models)

        Returns:
            URL string or None
        """
        if model_id not in AVAILABLE_MODELS:
            return None

        info = AVAILABLE_MODELS[model_id]
        repo = info["huggingface_repo"]

        if info["type"] == "llm":
            # For LLM, return direct link to GGUF file
            gguf_file = info.get("filename", "")
            return f"https://huggingface.co/{repo}/resolve/main/{gguf_file}"
        else:
            # For embedding models, return link to specific file or repo
            if filename:
                return f"https://huggingface.co/{repo}/resolve/main/{filename}"
            return f"https://huggingface.co/{repo}"

    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get detailed info about a specific model"""
        if model_id not in AVAILABLE_MODELS:
            return None

        info = AVAILABLE_MODELS[model_id].copy()
        info["id"] = model_id
        info["installed"] = self.is_model_installed(model_id)
        info["path"] = str(self.get_model_path(model_id)) if self.get_model_path(model_id) else None

        return info

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for model recommendations"""
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024 ** 3)

        # Get available disk space in models folder
        models_folder = self._get_models_folder()
        if models_folder and models_folder.exists():
            disk_free_gb = psutil.disk_usage(str(models_folder)).free / (1024 ** 3)
        elif self.backup_folder:
            disk_free_gb = psutil.disk_usage(self.backup_folder).free / (1024 ** 3)
        else:
            disk_free_gb = 0

        # Determine recommended models based on RAM
        if ram_gb < 4:
            recommended_llm = "tinyllama-1.1b-q4"
            recommended_embedding = "all-MiniLM-L6-v2"  # 384-dim, 80MB
            tier = "constrained"
        elif ram_gb < 8:
            recommended_llm = "llama-3.2-3b-q4"
            recommended_embedding = "all-mpnet-base-v2"  # 768-dim, 420MB
            tier = "standard"
        else:
            recommended_llm = "mistral-7b-q4"
            recommended_embedding = "intfloat-e5-large-v2"  # 1024-dim, 1.3GB
            tier = "capable"

        return {
            "ram_gb": round(ram_gb, 1),
            "disk_free_gb": round(disk_free_gb, 1),
            "tier": tier,
            "recommended_embedding": recommended_embedding,
            "recommended_llm": recommended_llm
        }


# Singleton instance
_registry_instance = None


def get_model_registry(backup_folder: Optional[str] = None) -> ModelRegistry:
    """Get or create the singleton registry instance"""
    global _registry_instance
    if _registry_instance is None or backup_folder:
        _registry_instance = ModelRegistry(backup_folder)
    return _registry_instance


def reload_model_registry() -> ModelRegistry:
    """Force reload of the registry"""
    global _registry_instance
    _registry_instance = None
    return get_model_registry()
