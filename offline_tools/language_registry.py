"""
Language Pack Registry for Offline Translation
Manages translation model packs for offline article translation.

Supports MarianMT per-language packs (Phase 1-2) and NLLB universal (Phase 3).
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


# Available language packs with metadata
# Downloads directly from HuggingFace - no R2 hosting needed
AVAILABLE_LANGUAGE_PACKS = {
    "es": {
        "type": "marian",
        "display_name": "Spanish",
        "native_name": "Espanol",
        "direction": "ltr",
        "size_mb": 300,
        "min_ram_gb": 2,
        "huggingface_repos": {
            "en_to_lang": "Helsinki-NLP/opus-mt-en-es",
            "lang_to_en": "Helsinki-NLP/opus-mt-es-en"
        },
        "description": "Translation between English and Spanish"
    },
    "fr": {
        "type": "marian",
        "display_name": "French",
        "native_name": "Francais",
        "direction": "ltr",
        "size_mb": 290,
        "min_ram_gb": 2,
        "huggingface_repos": {
            "en_to_lang": "Helsinki-NLP/opus-mt-en-fr",
            "lang_to_en": "Helsinki-NLP/opus-mt-fr-en"
        },
        "description": "Translation between English and French"
    },
    "ar": {
        "type": "marian",
        "display_name": "Arabic",
        "native_name": "العربية",
        "direction": "rtl",
        "size_mb": 320,
        "min_ram_gb": 2,
        "huggingface_repos": {
            "en_to_lang": "Helsinki-NLP/opus-mt-en-ar",
            "lang_to_en": "Helsinki-NLP/opus-mt-ar-en"
        },
        "description": "Translation between English and Arabic"
    },
    "zh": {
        "type": "marian",
        "display_name": "Chinese (Simplified)",
        "native_name": "简体中文",
        "direction": "ltr",
        "size_mb": 350,
        "min_ram_gb": 2,
        "huggingface_repos": {
            "en_to_lang": "Helsinki-NLP/opus-mt-en-zh",
            "lang_to_en": "Helsinki-NLP/opus-mt-zh-en"
        },
        "description": "Translation between English and Simplified Chinese"
    },
    "pt": {
        "type": "marian",
        "display_name": "Portuguese",
        "native_name": "Portugues",
        "direction": "ltr",
        "size_mb": 290,
        "min_ram_gb": 2,
        "huggingface_repos": {
            "en_to_lang": "Helsinki-NLP/opus-mt-en-pt",
            "lang_to_en": "Helsinki-NLP/opus-mt-pt-en"
        },
        "description": "Translation between English and Portuguese"
    },
    "hi": {
        "type": "marian",
        "display_name": "Hindi",
        "native_name": "हिन्दी",
        "direction": "ltr",
        "size_mb": 300,
        "min_ram_gb": 2,
        "huggingface_repos": {
            "en_to_lang": "Helsinki-NLP/opus-mt-en-hi",
            "lang_to_en": "Helsinki-NLP/opus-mt-hi-en"
        },
        "description": "Translation between English and Hindi"
    },
    "sw": {
        "type": "marian",
        "display_name": "Swahili",
        "native_name": "Kiswahili",
        "direction": "ltr",
        "size_mb": 280,
        "min_ram_gb": 2,
        "huggingface_repos": {
            "en_to_lang": "Helsinki-NLP/opus-mt-en-swc",
            "lang_to_en": "Helsinki-NLP/opus-mt-swc-en"
        },
        "description": "Translation between English and Swahili"
    },
    "ht": {
        "type": "marian",
        "display_name": "Haitian Creole",
        "native_name": "Kreyol Ayisyen",
        "direction": "ltr",
        "size_mb": 280,
        "min_ram_gb": 2,
        "huggingface_repos": {
            "en_to_lang": "Helsinki-NLP/opus-mt-en-ht",
            "lang_to_en": "Helsinki-NLP/opus-mt-ht-en"
        },
        "description": "Translation between English and Haitian Creole"
    }
}


class LanguageRegistry:
    """Manages installed language packs and their configuration"""

    DEFAULT_REGISTRY = {
        "schema_version": 1,
        "last_updated": None,
        "active_language": "en",
        "installed_packs": []
    }

    def __init__(self, backup_folder: Optional[str] = None):
        """
        Initialize language registry.

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

    def _get_translation_folder(self) -> Optional[Path]:
        """Get the translation models folder path"""
        if not self.backup_folder:
            return None
        return Path(self.backup_folder) / "models" / "translation"

    def _get_registry_path(self) -> Optional[Path]:
        """Get path to _languages.json registry file"""
        translation_folder = self._get_translation_folder()
        if not translation_folder:
            return None
        return translation_folder / "_languages.json"

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
                print(f"Error loading language registry: {e}")
        return self.DEFAULT_REGISTRY.copy()

    def save(self) -> bool:
        """Save registry to file"""
        registry_path = self._get_registry_path()
        if not registry_path:
            print("Cannot save registry: backup folder not configured")
            return False

        try:
            # Ensure folder exists
            registry_path.parent.mkdir(parents=True, exist_ok=True)

            self.registry["last_updated"] = datetime.now().isoformat()
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2)
            return True
        except IOError as e:
            print(f"Error saving language registry: {e}")
            return False

    def get_pack_path(self, lang_code: str) -> Optional[Path]:
        """
        Get path to a specific language pack's folder.

        Args:
            lang_code: Language code (e.g., "es", "fr")

        Returns:
            Path to language pack folder or None
        """
        if lang_code not in AVAILABLE_LANGUAGE_PACKS:
            return None

        translation_folder = self._get_translation_folder()
        if not translation_folder:
            return None

        pack_info = AVAILABLE_LANGUAGE_PACKS[lang_code]
        if pack_info["type"] == "marian":
            return translation_folder / "marian" / lang_code

        return None

    def is_pack_installed(self, lang_code: str) -> bool:
        """Check if a language pack is installed locally"""
        pack_path = self.get_pack_path(lang_code)
        if not pack_path or not pack_path.exists():
            return False

        pack_info = AVAILABLE_LANGUAGE_PACKS.get(lang_code)
        if not pack_info:
            return False

        # Check for required model directories
        if pack_info["type"] == "marian":
            repos = pack_info.get("huggingface_repos", {})
            en_to_lang_name = repos.get("en_to_lang", "").split("/")[-1]
            lang_to_en_name = repos.get("lang_to_en", "").split("/")[-1]

            en_to_lang_path = pack_path / en_to_lang_name
            # For Phase 1, only en_to_lang is required (article translation)
            # lang_to_en is needed for Phase 2 (chat translation)
            if en_to_lang_path.exists():
                # Check for model file
                return (en_to_lang_path / "pytorch_model.bin").exists() or \
                       (en_to_lang_path / "model.safetensors").exists()

        return False

    def get_installed_packs(self) -> List[str]:
        """Get list of installed language pack codes"""
        installed = []
        for lang_code in AVAILABLE_LANGUAGE_PACKS.keys():
            if self.is_pack_installed(lang_code):
                installed.append(lang_code)
        return installed

    def get_active_language(self) -> str:
        """Get the currently active language code"""
        return self.registry.get("active_language", "en")

    def set_active_language(self, lang_code: str) -> bool:
        """
        Set the active language for translation.

        Args:
            lang_code: Language code ("en" for no translation, or installed pack code)

        Returns:
            True if successful
        """
        if lang_code != "en" and lang_code not in AVAILABLE_LANGUAGE_PACKS:
            return False

        if lang_code != "en" and not self.is_pack_installed(lang_code):
            return False

        self.registry["active_language"] = lang_code
        return self.save()

    def get_available_packs(self) -> Dict[str, Dict]:
        """
        Get all available language packs with their install status.

        Returns:
            Dict of lang_code -> pack info with "installed" field added
        """
        result = {}
        for lang_code, info in AVAILABLE_LANGUAGE_PACKS.items():
            pack_data = info.copy()
            pack_data["code"] = lang_code
            pack_data["installed"] = self.is_pack_installed(lang_code)
            pack_data["active"] = (lang_code == self.registry.get("active_language"))

            result[lang_code] = pack_data

        return result

    def get_pack_info(self, lang_code: str) -> Optional[Dict]:
        """Get detailed info about a specific language pack"""
        if lang_code not in AVAILABLE_LANGUAGE_PACKS:
            return None

        info = AVAILABLE_LANGUAGE_PACKS[lang_code].copy()
        info["code"] = lang_code
        info["installed"] = self.is_pack_installed(lang_code)
        info["active"] = (lang_code == self.registry.get("active_language"))
        info["path"] = str(self.get_pack_path(lang_code)) if self.get_pack_path(lang_code) else None

        return info

    def mark_pack_installed(self, lang_code: str) -> bool:
        """
        Mark a language pack as installed in the registry.

        Args:
            lang_code: Language code that was installed

        Returns:
            True if saved successfully
        """
        installed = self.registry.get("installed_packs", [])
        if lang_code not in installed:
            installed.append(lang_code)
            self.registry["installed_packs"] = installed
        return self.save()

    def mark_pack_removed(self, lang_code: str) -> bool:
        """
        Mark a language pack as removed from the registry.

        Args:
            lang_code: Language code that was removed

        Returns:
            True if saved successfully
        """
        installed = self.registry.get("installed_packs", [])
        if lang_code in installed:
            installed.remove(lang_code)
            self.registry["installed_packs"] = installed

        # Reset active language if the removed pack was active
        if self.registry.get("active_language") == lang_code:
            self.registry["active_language"] = "en"

        return self.save()

    def get_huggingface_url(self, lang_code: str, direction: str = "en_to_lang") -> Optional[str]:
        """
        Get HuggingFace download URL for a language pack model.

        Args:
            lang_code: Language code
            direction: "en_to_lang" or "lang_to_en"

        Returns:
            HuggingFace repo URL or None
        """
        if lang_code not in AVAILABLE_LANGUAGE_PACKS:
            return None

        info = AVAILABLE_LANGUAGE_PACKS[lang_code]
        repos = info.get("huggingface_repos", {})
        repo = repos.get(direction)

        if repo:
            return f"https://huggingface.co/{repo}"
        return None


# Singleton instance
_registry_instance = None


def get_language_registry(backup_folder: Optional[str] = None) -> LanguageRegistry:
    """Get or create the singleton registry instance"""
    global _registry_instance
    if _registry_instance is None or backup_folder:
        _registry_instance = LanguageRegistry(backup_folder)
    return _registry_instance


def reload_language_registry() -> LanguageRegistry:
    """Force reload of the registry"""
    global _registry_instance
    _registry_instance = None
    return get_language_registry()
