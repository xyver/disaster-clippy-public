"""
Translation Service for Offline Language Support
Supports MarianMT (per-language) models for article translation.

Phase 1: Article translation only (translate_from_english)
Phase 2: Add chat translation (translate_to_english)
Phase 3: Add NLLB universal model support
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import hashlib


class TranslationService:
    """Handles translation for offline article viewing"""

    def __init__(self, language_code: str = None):
        """
        Initialize translation service.

        Args:
            language_code: Target language code. If None, gets from config.
        """
        self.language = language_code or self._get_active_language()
        self.cache = TranslationCache()
        self._model_en_to_lang = None
        self._tokenizer_en_to_lang = None
        self._model_lang_to_en = None  # For Phase 2
        self._tokenizer_lang_to_en = None  # For Phase 2
        self._model_loaded = False
        self._device = None
        self._gpu_count = 0
        self._batch_size = self._get_batch_size()

    def _get_batch_size(self) -> int:
        """Get batch size from config (for UI configurability)"""
        try:
            from admin.local_config import get_local_config
            config = get_local_config()
            return config.get("translation.batch_size", 256)
        except Exception:
            return 256  # Default for GPU

    def _get_active_language(self) -> str:
        """Get active language from config"""
        try:
            from admin.local_config import get_local_config
            config = get_local_config()
            return config.get("translation.active_language", "en")
        except Exception:
            return "en"

    def _get_backup_folder(self) -> Optional[str]:
        """Get backup folder path"""
        try:
            from admin.local_config import get_local_config
            return get_local_config().get_backup_folder()
        except Exception:
            return os.getenv("BACKUP_PATH")

    def is_available(self) -> bool:
        """Check if translation is available for current language"""
        if self.language == "en":
            return False  # No translation needed

        from .language_registry import get_language_registry
        registry = get_language_registry()
        return registry.is_pack_installed(self.language)

    def _load_model(self) -> bool:
        """
        Load the MarianMT model for translation.
        Automatically uses GPU if available, including multi-GPU with DataParallel.

        Returns:
            True if model loaded successfully
        """
        if self._model_loaded:
            return True

        if self.language == "en":
            return False

        try:
            import torch
            from transformers import MarianMTModel, MarianTokenizer
            from .language_registry import get_language_registry, AVAILABLE_LANGUAGE_PACKS

            # Detect GPU/CUDA
            if torch.cuda.is_available():
                self._gpu_count = torch.cuda.device_count()
                self._device = torch.device("cuda")
                gpu_names = [torch.cuda.get_device_name(i) for i in range(self._gpu_count)]
                print(f"[Translation] CUDA available: {self._gpu_count} GPU(s) detected")
                for i, name in enumerate(gpu_names):
                    vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"  GPU {i}: {name} ({vram:.1f} GB)")
            else:
                self._device = torch.device("cpu")
                print("[Translation] CUDA not available, using CPU")

            registry = get_language_registry()
            pack_path = registry.get_pack_path(self.language)

            if not pack_path or not pack_path.exists():
                print(f"Language pack not installed: {self.language}")
                return False

            pack_info = AVAILABLE_LANGUAGE_PACKS.get(self.language)
            if not pack_info:
                return False

            repos = pack_info.get("huggingface_repos", {})

            # Load English -> Language model (for article translation)
            en_to_lang_name = repos.get("en_to_lang", "").split("/")[-1]
            en_to_lang_path = pack_path / en_to_lang_name

            if en_to_lang_path.exists():
                print(f"[Translation] Loading model from {en_to_lang_path}...")
                self._tokenizer_en_to_lang = MarianTokenizer.from_pretrained(str(en_to_lang_path))
                self._model_en_to_lang = MarianMTModel.from_pretrained(str(en_to_lang_path))

                # Move model to GPU
                self._model_en_to_lang = self._model_en_to_lang.to(self._device)

                # Use half-precision (float16) only on GPUs with Tensor Cores (RTX 20xx+)
                # Pascal (GTX 10xx) has compute capability 6.x - no tensor cores
                # Turing+ (RTX 20xx+) has compute capability 7.5+ - has tensor cores
                if self._device.type == "cuda":
                    compute_cap = torch.cuda.get_device_capability(0)
                    if compute_cap[0] >= 7 and compute_cap[1] >= 5:
                        self._model_en_to_lang = self._model_en_to_lang.half()
                        print(f"[Translation] Using float16 (GPU compute {compute_cap[0]}.{compute_cap[1]} has Tensor Cores)")
                    else:
                        print(f"[Translation] Using float32 (GPU compute {compute_cap[0]}.{compute_cap[1]} - no Tensor Cores)")

                # Use DataParallel for multi-GPU
                if self._gpu_count > 1:
                    print(f"[Translation] Enabling DataParallel across {self._gpu_count} GPUs")
                    self._model_en_to_lang = torch.nn.DataParallel(self._model_en_to_lang)

                # Set to evaluation mode (faster inference)
                self._model_en_to_lang.eval()

                self._model_loaded = True
                device_str = f"GPU x{self._gpu_count}" if self._gpu_count > 0 else "CPU"
                print(f"[Translation] Model loaded for {self.language} on {device_str}")
                return True
            else:
                print(f"Model path not found: {en_to_lang_path}")
                return False

        except ImportError as e:
            print(f"Missing dependency for translation: {e}")
            print("Install with: pip install transformers sentencepiece sacremoses")
            return False
        except Exception as e:
            print(f"Error loading translation model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _should_translate(self, text: str) -> bool:
        """
        Check if text should be translated (skip numbers, dates, single words, etc.)
        """
        import re

        # Skip if mostly numbers/punctuation
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count < len(text) * 0.5:
            return False

        # Skip single words (likely proper nouns, UI labels)
        words = text.split()
        if len(words) < 2:
            return False

        # Skip if looks like a date, version, or technical string
        if re.match(r'^[\d\-\./:\s]+$', text):
            return False
        if re.match(r'^v?\d+(\.\d+)+', text):  # Version numbers
            return False

        return True

    def translate_from_english(self, text: str) -> str:
        """
        Translate English text to the target language.

        Args:
            text: English text to translate

        Returns:
            Translated text, or original if translation fails
        """
        if self.language == "en" or not text.strip():
            return text

        if not self._load_model():
            return text

        try:
            max_length = 512
            inputs = self._tokenizer_en_to_lang(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            outputs = self._model_en_to_lang.generate(**inputs)
            translated = self._tokenizer_en_to_lang.decode(outputs[0], skip_special_tokens=True)
            return translated

        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def translate_batch(self, texts: list) -> list:
        """
        Translate multiple texts in a single batch (much faster than individual calls).
        Automatically uses GPU if available.

        Args:
            texts: List of English texts to translate

        Returns:
            List of translated texts
        """
        if self.language == "en" or not texts:
            return texts

        if not self._load_model():
            return texts

        try:
            import torch

            # Filter out empty texts but track positions
            # Also truncate very long texts to speed up translation
            max_chars = 500  # Truncate texts longer than this
            non_empty = []
            for i, t in enumerate(texts):
                if t and t.strip():
                    # Truncate long texts
                    truncated = t[:max_chars] + "..." if len(t) > max_chars else t
                    non_empty.append((i, truncated))

            if not non_empty:
                return texts

            indices, to_translate = zip(*non_empty)

            # Batch translate with optimizations
            max_length = 256  # Reduced for speed
            inputs = self._tokenizer_en_to_lang(
                list(to_translate),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )

            # Move inputs to GPU if available
            if self._device is not None:
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Get the actual model (unwrap DataParallel if needed)
            model = self._model_en_to_lang
            if hasattr(model, 'module'):
                # DataParallel wraps the model in .module
                generate_model = model.module
            else:
                generate_model = model

            # Disable gradient computation for faster inference
            with torch.no_grad():
                outputs = generate_model.generate(
                    **inputs,
                    max_new_tokens=256,  # Limit output length
                    num_beams=1,  # Greedy decoding (faster than beam search)
                    do_sample=False
                )

            # Decode all outputs
            translated = self._tokenizer_en_to_lang.batch_decode(outputs, skip_special_tokens=True)

            # Rebuild result list with translations in correct positions
            result = list(texts)  # Start with originals
            for idx, trans in zip(indices, translated):
                result[idx] = trans

            return result

        except Exception as e:
            print(f"Batch translation error: {e}")
            import traceback
            traceback.print_exc()
            return texts

    def translate_html(self, html: str, article_id: str = None) -> str:
        """
        Translate HTML content preserving tags.
        Uses batch translation for much better performance.

        Args:
            html: HTML content to translate
            article_id: Optional article ID for caching

        Returns:
            Translated HTML
        """
        if self.language == "en":
            return html

        # Check cache first
        if article_id:
            cached = self.cache.get_article(article_id, self.language)
            if cached:
                print(f"[Translation] Using cached translation for {article_id}")
                return cached

        if not self._load_model():
            return html

        try:
            from bs4 import BeautifulSoup
            import time

            t0 = time.time()
            # Use lxml parser (5-10x faster than html.parser)
            soup = BeautifulSoup(html, 'lxml')

            # Tags to skip (don't translate code, scripts, metadata, etc.)
            skip_tags = {'script', 'style', 'code', 'pre', 'kbd', 'var', 'noscript',
                         'cite', 'sup', 'sub', 'nav', 'footer', 'header'}

            # Classes to skip (references, navigation, metadata boxes)
            skip_classes = {'reference', 'references', 'reflist', 'refbegin',
                           'mw-editsection', 'mw-headline-anchor', 'cite-bracket',
                           'toc', 'catlinks', 'navbox', 'sistersitebox',
                           'mbox', 'ambox', 'metadata', 'noprint', 'plainlinks',
                           'infobox', 'sidebar', 'vertical-navbox'}

            def should_skip_element(elem):
                """Check if element or any parent should be skipped"""
                for parent in elem.parents:
                    if parent.name in skip_tags:
                        return True
                    parent_classes = parent.get('class', [])
                    if any(c in skip_classes for c in parent_classes):
                        return True
                return False

            # Collect all text nodes that need translation
            text_nodes = []
            texts_to_translate = []

            for text_node in soup.find_all(string=True):
                if text_node.parent.name in skip_tags:
                    continue

                if should_skip_element(text_node):
                    continue

                original = str(text_node).strip()
                # Skip short texts, numbers, dates, and non-prose content
                if original and len(original) > 3 and self._should_translate(original):
                    text_nodes.append(text_node)
                    texts_to_translate.append(original)

            if not texts_to_translate:
                return html

            parse_time = time.time() - t0
            print(f"[Translation] Parsed {len(texts_to_translate)} segments in {parse_time:.2f}s")

            # Batch translate - configurable via translation.batch_size in settings
            # GPU can handle 256+ with 8GB VRAM, CPU is memory-limited
            batch_size = self._batch_size if self._gpu_count > 0 else min(self._batch_size, 64)
            all_translated = []
            total_batches = (len(texts_to_translate) + batch_size - 1) // batch_size

            t1 = time.time()
            for i in range(0, len(texts_to_translate), batch_size):
                batch_num = i // batch_size + 1
                batch = texts_to_translate[i:i + batch_size]
                translated_batch = self.translate_batch(batch)
                all_translated.extend(translated_batch)

            translate_time = time.time() - t1
            print(f"[Translation] Translated {len(texts_to_translate)} texts in {translate_time:.2f}s ({translate_time/len(texts_to_translate)*1000:.0f}ms/text)")

            # Replace text nodes with translations
            for text_node, translated in zip(text_nodes, all_translated):
                text_node.replace_with(translated)

            result = str(soup)

            # Cache the result
            if article_id:
                self.cache.save_article(article_id, self.language, result)
                print(f"[Translation] Cached translation for {article_id}")

            return result

        except ImportError:
            print("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            return html
        except Exception as e:
            print(f"HTML translation error: {e}")
            return html

    def get_status(self) -> Dict[str, Any]:
        """Get translation service status"""
        return {
            "language": self.language,
            "available": self.is_available(),
            "model_loaded": self._model_loaded,
            "cache_enabled": self.cache.is_enabled()
        }


class TranslationCache:
    """Manages cached article translations"""

    def __init__(self, backup_folder: str = None):
        """
        Initialize translation cache.

        Args:
            backup_folder: Path to backup folder. Auto-detected if None.
        """
        self.backup_folder = backup_folder
        if not self.backup_folder:
            try:
                from admin.local_config import get_local_config
                self.backup_folder = get_local_config().get_backup_folder()
            except Exception:
                self.backup_folder = os.getenv("BACKUP_PATH")

    def _get_cache_path(self) -> Optional[Path]:
        """Get the translations cache folder path"""
        if not self.backup_folder:
            return None
        return Path(self.backup_folder) / "translations"

    def is_enabled(self) -> bool:
        """Check if caching is enabled"""
        try:
            from admin.local_config import get_local_config
            return get_local_config().get("translation.cache_enabled", True)
        except Exception:
            return True

    def get_article(self, article_id: str, language: str) -> Optional[str]:
        """
        Get cached article translation.

        Args:
            article_id: Article identifier (e.g., "appropedia/article_123")
            language: Target language code

        Returns:
            Cached translated content or None
        """
        if not self.is_enabled():
            return None

        cache_path = self._get_cache_path()
        if not cache_path:
            return None

        # Sanitize article_id for filename
        safe_id = article_id.replace("/", "_").replace("\\", "_")
        cache_file = cache_path / language / "articles" / f"{safe_id}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("content")
            except (json.JSONDecodeError, IOError):
                pass

        return None

    def save_article(self, article_id: str, language: str, content: str) -> bool:
        """
        Save translated article to cache.

        Args:
            article_id: Article identifier
            language: Target language code
            content: Translated HTML content

        Returns:
            True if saved successfully
        """
        if not self.is_enabled():
            return False

        cache_path = self._get_cache_path()
        if not cache_path:
            return False

        try:
            # Sanitize article_id for filename
            safe_id = article_id.replace("/", "_").replace("\\", "_")
            cache_file = cache_path / language / "articles" / f"{safe_id}.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "article_id": article_id,
                "language": language,
                "translated_at": datetime.now().isoformat(),
                "content_hash": hashlib.sha256(content.encode()).hexdigest()[:16],
                "content": content
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)

            return True
        except IOError as e:
            print(f"Error saving translation cache: {e}")
            return False

    def clear_language(self, language: str) -> int:
        """
        Clear all cached translations for a language.

        Args:
            language: Language code to clear

        Returns:
            Number of files deleted
        """
        cache_path = self._get_cache_path()
        if not cache_path:
            return 0

        lang_cache = cache_path / language
        if not lang_cache.exists():
            return 0

        count = 0
        try:
            import shutil
            for item in lang_cache.rglob("*.json"):
                item.unlink()
                count += 1
        except Exception as e:
            print(f"Error clearing cache: {e}")

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_path = self._get_cache_path()
        if not cache_path or not cache_path.exists():
            return {
                "enabled": self.is_enabled(),
                "size_mb": 0,
                "article_count": 0,
                "languages": []
            }

        total_size = 0
        article_count = 0
        languages = []

        try:
            for lang_dir in cache_path.iterdir():
                if lang_dir.is_dir():
                    languages.append(lang_dir.name)
                    articles_dir = lang_dir / "articles"
                    if articles_dir.exists():
                        for f in articles_dir.glob("*.json"):
                            total_size += f.stat().st_size
                            article_count += 1
        except Exception:
            pass

        return {
            "enabled": self.is_enabled(),
            "size_mb": round(total_size / (1024 * 1024), 2),
            "article_count": article_count,
            "languages": languages
        }


# Singleton instance
_translation_service = None


def get_translation_service(language: str = None) -> TranslationService:
    """
    Get or create translation service.

    Args:
        language: Target language. If None, uses active language from config.

    Returns:
        TranslationService instance
    """
    global _translation_service

    # Get target language
    if language is None:
        try:
            from admin.local_config import get_local_config
            language = get_local_config().get("translation.active_language", "en")
        except Exception:
            language = "en"

    # Return existing instance if language matches
    if _translation_service is not None and _translation_service.language == language:
        return _translation_service

    # Create new instance
    _translation_service = TranslationService(language)
    return _translation_service


def translate_article(article_id: str, html: str, language: str = None) -> str:
    """
    Convenience function to translate an article.

    Args:
        article_id: Article identifier for caching
        html: HTML content to translate
        language: Target language (None = use active)

    Returns:
        Translated HTML
    """
    service = get_translation_service(language)
    return service.translate_html(html, article_id)
