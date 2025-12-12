# Language Packs - Offline Translation

This document covers the implementation of language packs for offline translation in the disaster preparedness app. It is self-contained with all relevant code sections, architecture decisions, and implementation details.

**Goal:** Enable non-English users to interact with the app in their language, with all content translated on-device.

**Prerequisite:** For full AI experience, users should also have the Embedding Model and LLM Model installed (see [offline-upgrade.md](offline-upgrade.md)).

---

## Table of Contents
1. [System Architecture Context](#system-architecture-context)
2. [Core Concept](#core-concept)
3. [Translation Flow](#translation-flow)
4. [Why English-Only Index](#why-english-only-index)
5. [Translation Model Options](#translation-model-options)
6. [Language Pack Structure](#language-pack-structure)
7. [Translation Caching Strategy](#translation-caching-strategy)
8. [UI Translation](#ui-translation)
9. [Translation Service Code](#translation-service-code)
10. [Chat Integration](#chat-integration)
11. [Website Viewer Integration](#website-viewer-integration)
12. [Priority Languages](#priority-languages)
13. [Implementation Phases](#implementation-phases)
14. [Size Estimates](#size-estimates)
15. [Three-Pack Model System](#three-pack-model-system)

---

## System Architecture Context

### Dual Embedding Architecture

The system uses two embedding dimensions:

| Context | Dimension | Model | Purpose |
|---------|-----------|-------|---------|
| Online (Pinecone) | 1536 | OpenAI text-embedding-3-small | Cloud search |
| Offline (local ChromaDB) | 768 | sentence-transformers | Local search |

**All content is indexed in English.** Translation happens at query time and display time.

### User Tiers

| Tier | Hardware | What They Do |
|------|----------|--------------|
| **Consumer** | RPi5 / Field device | Download packs, search, browse offline |
| **Local Admin** | Laptop 8-16GB | Create sources, submit to global |
| **Global Admin** | Desktop + API | Review, standardize, publish |

### Where Translation Fits

Translation is a **consumer-tier feature** that works alongside:
1. **Embedding Model** (768-dim) - For semantic search
2. **LLM Model** (Llama 3.2 3B) - For AI responses
3. **Language Pack** - For non-English interaction

```
User Experience Stack (Non-English User):
+----------------------------+
| Language Pack (~300MB)     |  <-- Translates input/output
+----------------------------+
| LLM Model (~2GB)           |  <-- Generates responses (in English)
+----------------------------+
| Embedding Model (~420MB)   |  <-- Semantic search (English vectors)
+----------------------------+
| Source Packs (768-dim)     |  <-- English content with vectors
+----------------------------+
```

### Key Files in Codebase

| File | Purpose |
|------|---------|
| `offline_tools/embeddings.py` | Current embedding service |
| `admin/ai_service.py` | Chat/search orchestration |
| `admin/routes/packs.py` | Download system |
| `local_settings.json` | User configuration |
| `offline_tools/translation.py` | **NEW: Translation service** |

---

## Core Concept

All content is indexed and stored in English. Language packs enable:
1. **Chat translation**: User speaks any language, system translates to/from English
2. **Website translation**: Offline articles displayed in user's language
3. **UI translation**: App interface in user's language

---

## Translation Flow

### Chat/Search Flow

```
USER INPUT (Spanish)
    |
    v
Translate to English (Translation Model)
    |
    v
Embed English query (Embedding Model, 768-dim)
    |
    v
Search ChromaDB (English vectors)
    |
    v
Retrieve English documents
    |
    v
LLM generates English response
    |
    v
Translate to Spanish (Translation Model)
    |
    v
USER SEES SPANISH RESPONSE
```

### Website Browsing Flow

```
User clicks link -> Load English HTML -> Translate -> Cache -> Display in Spanish
```

---

## Why English-Only Index

| Approach | Pros | Cons |
|----------|------|------|
| **English-only index** | One vector DB, consistent embeddings, smaller storage | Requires translation model |
| Multi-language index | No translation needed | Multiple vector DBs, inconsistent quality, huge storage |

**Decision: English-only index with translation layer**

This means:
- All source packs stored in English
- All vectors generated from English text
- Translation happens at query time and display time
- Users can still search/chat in any language

---

## Translation Model Options

Two approaches available:

| Model | Languages | Size | Best For |
|-------|-----------|------|----------|
| **MarianMT (per pair)** | 2 (en<->X) | ~300MB per language | Home users, single language |
| **NLLB-200-distilled-600M** | 200 (any direction) | ~2.4GB total | Community centers, multi-language |

### Break-Even Analysis

| Languages Needed | MarianMT Total | NLLB Total | Winner |
|------------------|----------------|------------|--------|
| 1 language | 300MB | 2.4GB | MarianMT |
| 2 languages | 600MB | 2.4GB | MarianMT |
| 4 languages | 1.2GB | 2.4GB | MarianMT |
| 5+ languages | 1.5GB+ | 2.4GB | NLLB |

### Decision: Offer BOTH Options

**Option A: Single Language Packs (MarianMT)**
- Download: ~300MB per language (bidirectional)
- Best for: Home users, RPi5, single language needs
- Pros: Small, fast, optimized for specific pair
- Cons: Need separate download per language

**Option B: Universal Translator (NLLB)**
- Download: ~2.4GB once
- Best for: Community centers, NGOs, multi-lingual areas
- Pros: ALL 200 languages with one download, no additional downloads
- Cons: Larger initial download, slightly slower per-translation
- Hardware: Needs 8GB+ RAM (not suitable for RPi5 4GB)

### Typical Deployments

| Deployment | Hardware | Recommendation |
|------------|----------|----------------|
| Home user (Spanish speaker) | RPi5 8GB | MarianMT Spanish pack (300MB) |
| Family (Spanish + French) | RPi5 8GB | MarianMT both packs (600MB) |
| Community center | Server 16GB+ | NLLB Universal (2.4GB) |
| NGO field office | Laptop 8GB+ | NLLB Universal (2.4GB) |
| Refugee camp | Server 32GB+ | NLLB Universal (2.4GB) |

---

## Language Pack Structure

```
BACKUP_PATH/
|-- models/
|   |-- embeddings/          # Existing (see offline-upgrade.md)
|   |-- llm/                 # Existing (see offline-upgrade.md)
|   |-- translation/         # NEW
|       |-- _config.json     # Which mode: "marian" or "nllb"
|       |-- _languages.json  # Installed languages registry
|       |
|       |-- marian/          # Option A: Per-language packs
|       |   |-- es/          # Spanish (MarianMT)
|       |   |   |-- _manifest.json
|       |   |   |-- opus-mt-en-es/    # English -> Spanish
|       |   |   |   |-- config.json
|       |   |   |   |-- tokenizer_config.json
|       |   |   |   |-- pytorch_model.bin
|       |   |   |-- opus-mt-es-en/    # Spanish -> English
|       |   |       |-- ...
|       |   |-- fr/          # French
|       |   |-- ar/          # Arabic
|       |
|       |-- nllb/            # Option B: Universal model
|           |-- _manifest.json
|           |-- nllb-200-distilled-600M/
|               |-- config.json
|               |-- tokenizer files...
|               |-- pytorch_model.bin  # ~2.4GB
|
|-- translations/            # Cached translations (same for both modes)
    |-- es/
    |   |-- articles/        # Translated article cache
    |   |   |-- appropedia/
    |   |   |   |-- article_123.json
    |   |-- ui/              # UI string cache
    |       |-- strings.json
    |-- fr/
        |-- ...
```

### _config.json

```json
{
  "mode": "marian",
  "active_languages": ["es"],
  "nllb_installed": false
}
```

### Mode Switching

- If user has NLLB installed, can switch to "nllb" mode for any language
- If using MarianMT, limited to installed language packs
- System auto-detects available options and suggests best choice

---

## Language Pack Manifests

### _languages.json (Registry)

```json
{
  "schema_version": 1,
  "active_language": "es",
  "installed": {
    "es": {
      "name": "Spanish",
      "native_name": "Espanol",
      "size_bytes": 620000000,
      "installed_date": "2025-12-11T00:00:00Z"
    }
  },
  "ui_language": "es"
}
```

### Individual Language _manifest.json

```json
{
  "schema_version": 1,
  "language_code": "es",
  "language_name": "Spanish",
  "native_name": "Espanol",
  "direction": "ltr",

  "models": {
    "en_to_lang": "opus-mt-en-es",
    "lang_to_en": "opus-mt-es-en"
  },

  "size_bytes": 620000000,
  "runtime": "transformers",

  "requirements": {
    "min_ram_gb": 2,
    "recommended_ram_gb": 4
  },

  "source_url": "https://huggingface.co/Helsinki-NLP/opus-mt-en-es",
  "license": "Apache 2.0",
  "version": "1.0.0"
}
```

### NLLB _manifest.json

```json
{
  "schema_version": 1,
  "model_id": "nllb-200-distilled-600M",
  "model_type": "translation",
  "display_name": "NLLB Universal Translator",
  "description": "Translate between 200 languages with a single model",

  "languages": 200,
  "size_bytes": 2400000000,
  "runtime": "transformers",

  "requirements": {
    "min_ram_gb": 8,
    "recommended_ram_gb": 16
  },

  "source_url": "https://huggingface.co/facebook/nllb-200-distilled-600M",
  "license": "CC-BY-NC-4.0",
  "version": "1.0.0"
}
```

---

## Translation Caching Strategy

### Why Cache

- Translation is CPU-intensive
- Same articles viewed repeatedly
- Offline users have limited compute

### Cache Structure for Articles

```json
{
  "source_article_id": "appropedia/article_123",
  "source_hash": "sha256:abc123...",
  "language": "es",
  "translated_at": "2025-12-11T00:00:00Z",
  "sections": {
    "title": "Como construir un filtro de agua",
    "content": "Este articulo describe..."
  }
}
```

### Cache Invalidation

- If source article hash changes, re-translate
- User can force re-translate via UI
- Cache size limit with LRU eviction (configurable)

### Cache Settings (local_settings.json)

```json
{
  "translation": {
    "enabled": true,
    "active_language": "es",
    "cache_enabled": true,
    "cache_max_mb": 500,
    "cache_articles": true,
    "cache_ui": true
  }
}
```

---

## UI Translation

### Three-Tier Approach

| Tier | Method | Coverage | Speed |
|------|--------|----------|-------|
| 1. Built-in | Ship with app | Top 5-10 languages | Instant |
| 2. Cached | Translated once, saved | Any with language pack | Fast after first |
| 3. Live | Translate on demand | Fallback | Slow |

### Built-in UI Strings (Shipped with App)

```
app/
|-- locales/
    |-- en.json      # English (default)
    |-- es.json      # Spanish
    |-- fr.json      # French
    |-- zh.json      # Chinese
    |-- ar.json      # Arabic
```

### en.json Example

```json
{
  "nav": {
    "home": "Home",
    "search": "Search",
    "settings": "Settings",
    "sources": "Sources"
  },
  "search": {
    "placeholder": "Ask a question...",
    "no_results": "No results found",
    "semantic_search": "Semantic Search",
    "keyword_search": "Keyword Search"
  },
  "chat": {
    "thinking": "Thinking...",
    "sources_used": "Sources used:",
    "translate_response": "Translate response"
  }
}
```

### UI Translation Flow

```
App loads
    |
    v
Check user's language setting
    |
    +-- Built-in locale exists? --> Load it (instant)
    |
    +-- No built-in, but language pack installed?
            |
            v
        Check cached UI strings
            |
            +-- Cached? --> Load from cache
            |
            +-- Not cached? --> Translate all UI strings
                    |
                    v
                Save to cache
                    |
                    v
                Load translated UI
```

---

## Translation Service Code

### New Service: `offline_tools/translation.py`

```python
"""
Translation service for offline language support.
Supports MarianMT (per-language) and NLLB (universal).
"""

from typing import Dict, Optional
import os
import json
from pathlib import Path


class TranslationService:
    """Handles bidirectional translation for offline use"""

    def __init__(self, language_code: str = None):
        self.language = language_code or self._get_active_language()
        self.cache = TranslationCache()
        self._model_en_to_lang = None
        self._model_lang_to_en = None
        self._load_models()

    def _get_active_language(self) -> str:
        """Get active language from settings"""
        settings_path = self._get_settings_path()
        if settings_path.exists():
            with open(settings_path) as f:
                settings = json.load(f)
                return settings.get("translation", {}).get("active_language", "en")
        return "en"

    def _load_models(self):
        """Load translation models based on config"""
        if self.language == "en":
            return  # No translation needed

        config = self._load_config()
        mode = config.get("mode", "marian")

        if mode == "nllb":
            self._load_nllb_model()
        else:
            self._load_marian_models()

    def _load_marian_models(self):
        """Load MarianMT models for specific language pair"""
        from transformers import MarianMTModel, MarianTokenizer

        models_path = self._get_models_path() / "translation" / "marian" / self.language

        if not models_path.exists():
            raise ValueError(f"Language pack not installed: {self.language}")

        # Load English -> Language model
        en_to_lang_path = models_path / f"opus-mt-en-{self.language}"
        if en_to_lang_path.exists():
            self._tokenizer_en_to_lang = MarianTokenizer.from_pretrained(str(en_to_lang_path))
            self._model_en_to_lang = MarianMTModel.from_pretrained(str(en_to_lang_path))

        # Load Language -> English model
        lang_to_en_path = models_path / f"opus-mt-{self.language}-en"
        if lang_to_en_path.exists():
            self._tokenizer_lang_to_en = MarianTokenizer.from_pretrained(str(lang_to_en_path))
            self._model_lang_to_en = MarianMTModel.from_pretrained(str(lang_to_en_path))

    def _load_nllb_model(self):
        """Load NLLB universal translation model"""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_path = self._get_models_path() / "translation" / "nllb" / "nllb-200-distilled-600M"

        if not model_path.exists():
            raise ValueError("NLLB model not installed")

        self._nllb_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._nllb_model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))

    def translate_to_english(self, text: str) -> str:
        """Translate user input to English for search/LLM"""
        if self.language == "en":
            return text
        return self._translate(text, direction="to_en")

    def translate_from_english(self, text: str) -> str:
        """Translate English response to user's language"""
        if self.language == "en":
            return text
        return self._translate(text, direction="from_en")

    def _translate(self, text: str, direction: str) -> str:
        """Perform translation using loaded model"""
        config = self._load_config()
        mode = config.get("mode", "marian")

        if mode == "nllb":
            return self._translate_nllb(text, direction)
        else:
            return self._translate_marian(text, direction)

    def _translate_marian(self, text: str, direction: str) -> str:
        """Translate using MarianMT model"""
        if direction == "to_en":
            tokenizer = self._tokenizer_lang_to_en
            model = self._model_lang_to_en
        else:
            tokenizer = self._tokenizer_en_to_lang
            model = self._model_en_to_lang

        if model is None:
            raise ValueError(f"Translation model not loaded for direction: {direction}")

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _translate_nllb(self, text: str, direction: str) -> str:
        """Translate using NLLB model"""
        # NLLB uses language codes like "eng_Latn", "spa_Latn"
        lang_codes = {
            "en": "eng_Latn",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "ar": "arb_Arab",
            "zh": "zho_Hans",
            "pt": "por_Latn",
            "hi": "hin_Deva",
            "sw": "swh_Latn"
        }

        if direction == "to_en":
            src_lang = lang_codes.get(self.language, self.language)
            tgt_lang = "eng_Latn"
        else:
            src_lang = "eng_Latn"
            tgt_lang = lang_codes.get(self.language, self.language)

        self._nllb_tokenizer.src_lang = src_lang
        inputs = self._nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        translated = self._nllb_model.generate(
            **inputs,
            forced_bos_token_id=self._nllb_tokenizer.lang_code_to_id[tgt_lang]
        )

        return self._nllb_tokenizer.decode(translated[0], skip_special_tokens=True)

    def translate_article(self, article_id: str, html: str) -> str:
        """Translate article HTML, using cache if available"""
        cached = self.cache.get_article(article_id, self.language)
        if cached:
            return cached

        translated = self._translate_html(html)
        self.cache.save_article(article_id, self.language, translated)
        return translated

    def _translate_html(self, html: str) -> str:
        """Translate HTML content preserving tags"""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, 'html.parser')

        # Translate text nodes
        for text_node in soup.find_all(string=True):
            if text_node.parent.name not in ['script', 'style', 'code', 'pre']:
                original = str(text_node).strip()
                if original:
                    translated = self.translate_from_english(original)
                    text_node.replace_with(translated)

        return str(soup)

    def get_ui_strings(self) -> dict:
        """Get UI strings in user's language"""
        # Try built-in first
        builtin = self._load_builtin_locale()
        if builtin:
            return builtin

        # Try cache
        cached = self.cache.get_ui_strings(self.language)
        if cached:
            return cached

        # Translate and cache
        english = self._load_builtin_locale("en")
        translated = self._translate_dict(english)
        self.cache.save_ui_strings(self.language, translated)
        return translated

    def _translate_dict(self, data: dict) -> dict:
        """Recursively translate dictionary values"""
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = self._translate_dict(value)
            elif isinstance(value, str):
                result[key] = self.translate_from_english(value)
            else:
                result[key] = value
        return result

    def _load_builtin_locale(self, lang: str = None) -> Optional[dict]:
        """Load built-in locale file if exists"""
        lang = lang or self.language
        locale_path = Path(__file__).parent.parent / "app" / "locales" / f"{lang}.json"
        if locale_path.exists():
            with open(locale_path) as f:
                return json.load(f)
        return None

    def _load_config(self) -> dict:
        """Load translation config"""
        config_path = self._get_models_path() / "translation" / "_config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {"mode": "marian"}

    def _get_models_path(self) -> Path:
        """Get models folder path"""
        # Load from settings or use default
        return Path(os.getenv("BACKUP_PATH", ".")) / "models"

    def _get_settings_path(self) -> Path:
        """Get settings file path"""
        return Path("local_settings.json")


class TranslationCache:
    """Manages cached translations"""

    def __init__(self):
        self.cache_path = Path(os.getenv("BACKUP_PATH", ".")) / "translations"

    def get_article(self, article_id: str, language: str) -> Optional[str]:
        """Get cached article translation"""
        cache_file = self.cache_path / language / "articles" / f"{article_id.replace('/', '_')}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
                return data.get("content")
        return None

    def save_article(self, article_id: str, language: str, content: str):
        """Save article translation to cache"""
        cache_file = self.cache_path / language / "articles" / f"{article_id.replace('/', '_')}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        import hashlib
        data = {
            "source_article_id": article_id,
            "language": language,
            "translated_at": self._now(),
            "content": content
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_ui_strings(self, language: str) -> Optional[dict]:
        """Get cached UI strings"""
        cache_file = self.cache_path / language / "ui" / "strings.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def save_ui_strings(self, language: str, strings: dict):
        """Save UI strings to cache"""
        cache_file = self.cache_path / language / "ui" / "strings.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_file, 'w') as f:
            json.dump(strings, f, indent=2, ensure_ascii=False)

    def _now(self) -> str:
        """Get current ISO timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
```

---

## Chat Integration

### Updated Chat Flow

```python
def process_chat(user_message: str, user_language: str) -> str:
    """Process chat message with translation support"""
    translator = TranslationService(user_language)

    # Translate input if needed
    english_query = translator.translate_to_english(user_message)

    # Search (always in English)
    results = semantic_search(english_query)

    # Generate response (always in English)
    english_response = llm_generate(english_query, results)

    # Translate output if needed
    return translator.translate_from_english(english_response)
```

### Integration with Existing AIService

```python
# In admin/ai_service.py or similar

class AIService:
    def __init__(self):
        self.translator = None

    def set_language(self, language_code: str):
        """Set user's language for translation"""
        if language_code != "en":
            from offline_tools.translation import TranslationService
            self.translator = TranslationService(language_code)
        else:
            self.translator = None

    def search(self, query: str) -> list:
        """Search with translation support"""
        # Translate query to English if needed
        if self.translator:
            english_query = self.translator.translate_to_english(query)
        else:
            english_query = query

        # Perform search in English
        results = self._do_search(english_query)

        return results

    def generate_response(self, query: str, context: list) -> str:
        """Generate response with translation support"""
        # Translate query to English if needed
        if self.translator:
            english_query = self.translator.translate_to_english(query)
        else:
            english_query = query

        # Generate English response
        english_response = self._do_generate(english_query, context)

        # Translate response back if needed
        if self.translator:
            return self.translator.translate_from_english(english_response)
        else:
            return english_response
```

---

## Website Viewer Integration

### When User Views an Article

```python
def render_article(article_id: str, user_language: str) -> str:
    """Render article in user's language"""
    # Load English article
    html = load_article_html(article_id)

    if user_language == "en":
        return html

    # Translate (uses cache automatically)
    translator = TranslationService(user_language)
    return translator.translate_article(article_id, html)
```

### Flask Route Example

```python
@app.route('/view/<source_id>/<article_id>')
def view_article(source_id: str, article_id: str):
    """View article with translation"""
    user_language = get_user_language()  # From session/settings

    # Load article HTML
    article_path = get_article_path(source_id, article_id)
    with open(article_path) as f:
        html = f.read()

    # Translate if needed
    if user_language != "en":
        translator = TranslationService(user_language)
        html = translator.translate_article(f"{source_id}/{article_id}", html)

    return render_template('viewer.html', content=html)
```

---

## Priority Languages for Disaster Response

| Priority | Language | Why |
|----------|----------|-----|
| 1 | Spanish | Largest non-English population, Americas |
| 2 | French | Africa, Caribbean, Canada |
| 3 | Arabic | Middle East, North Africa |
| 4 | Chinese (Simplified) | Large population, disaster-prone regions |
| 5 | Portuguese | Brazil, Africa |
| 6 | Hindi | India, large population |
| 7 | Swahili | East Africa |
| 8 | Haitian Creole | Caribbean, high disaster risk |

**Start with Spanish** as proof of concept. Framework makes adding others trivial.

---

## Implementation Phases

### Phase 1: Translation Infrastructure

- [ ] Create TranslationService class
- [ ] Implement MarianMT model loading from portable path
- [ ] Basic translate_to/from_english methods
- [ ] Language detection

### Phase 2: Caching System

- [ ] TranslationCache class
- [ ] Article translation caching
- [ ] Cache invalidation logic
- [ ] LRU eviction when over size limit

### Phase 3: UI Translation

- [ ] Built-in locale files (en, es)
- [ ] UI string translation and caching
- [ ] Language selector in settings
- [ ] RTL support for Arabic

### Phase 4: Integration

- [ ] Chat service translation hooks
- [ ] Article viewer translation
- [ ] Search query translation
- [ ] Error messages in user's language

### Phase 5: Download UI

- [ ] Language pack cards in Models tab
- [ ] Download/install flow
- [ ] Active language selection
- [ ] NLLB vs MarianMT choice

### Phase 6: NLLB Support (Optional)

- [ ] NLLB model loading
- [ ] Language code mapping
- [ ] Mode switching in UI
- [ ] Auto-recommend based on language count

---

## Cloud Storage (R2)

```
r2://disaster-clippy-backups/
|-- models/
    |-- translation/
        |-- _registry.json         # Available options
        |-- marian/                 # Per-language packs
        |   |-- es.zip             # Spanish (~150MB compressed)
        |   |-- fr.zip             # French
        |   |-- ar.zip             # Arabic
        |   |-- zh.zip             # Chinese
        |-- nllb/                   # Universal model
            |-- nllb-200-distilled-600M.zip  # (~1.5GB compressed)
```

---

## Size Estimates

### Option A: MarianMT (per language)

| Language Pack | Compressed | Uncompressed |
|---------------|------------|--------------|
| Spanish (es) | ~150MB | ~300MB |
| French (fr) | ~140MB | ~290MB |
| Arabic (ar) | ~160MB | ~320MB |
| Chinese (zh) | ~175MB | ~350MB |
| **Per language avg** | **~150MB** | **~300MB** |

### Option B: NLLB Universal

| Model | Compressed | Uncompressed |
|-------|------------|--------------|
| NLLB-200-distilled-600M | ~1.5GB | ~2.4GB |

### Comparison for Multi-Language Needs

| Languages | MarianMT Total | NLLB Total |
|-----------|----------------|------------|
| 1 | 300MB | 2.4GB |
| 3 | 900MB | 2.4GB |
| 5 | 1.5GB | 2.4GB |
| 10 | 3GB | 2.4GB |
| 200 | 60GB (impractical) | 2.4GB |

Translation cache grows with use but is bounded by `cache_max_mb` setting.

---

## Three-Pack Model System

For non-English users, the complete offline experience requires three model packs:

```
REQUIRED FOR BASIC USE:
    None - keyword search works without any models

PRIORITY 1: Embedding Model (420MB)
    - Enables semantic search
    - Foundation for good results
    (See offline-upgrade.md for details)

PRIORITY 2: LLM Model (2GB)
    - Enables AI conversation
    - Enhancement over raw results
    (See offline-upgrade.md for details)

PRIORITY 3: Language Pack(s) (~300MB each)
    - Enables non-English use
    - Per-language download
```

### Full Offline Experience (Non-English User)

For a Spanish-speaking user on RPi5:

| Component | Size | Purpose |
|-----------|------|---------|
| Embedding Model | 420MB | Semantic search |
| LLM Model | 2GB | AI responses |
| Spanish Language Pack | ~620MB | Translation |
| Source Packs | varies | Content |
| **Total models** | **~3GB** | Full offline |

### User Experience Flow

1. Types question in Spanish
2. Question translated to English
3. Semantic search finds relevant English articles
4. LLM generates English response
5. Response translated to Spanish
6. User clicks article link
7. Article translated to Spanish (cached for next time)

---

## UI Design (Language Tab)

```
+------------------------------------------------------------------+
| Language Settings                                                |
+------------------------------------------------------------------+
|                                                                  |
| INTERFACE LANGUAGE                                               |
| ---------------------------------------------------------------- |
| Current: English                                                 |
| [English v]                                                      |
|                                                                  |
| TRANSLATION MODELS                                               |
| ---------------------------------------------------------------- |
|                                                                  |
| SINGLE LANGUAGE PACKS (Recommended for home use)                 |
| +---------------------------+  +---------------------------+     |
| | Spanish / Espanol        |  | French / Francais         |     |
| | Size: 300 MB             |  | Size: 290 MB              |     |
| | Status: Installed        |  | Status: Not installed     |     |
| |                          |  |                           |     |
| | [Active] [Remove]        |  | [Download]                |     |
| +---------------------------+  +---------------------------+     |
|                                                                  |
| +---------------------------+  +---------------------------+     |
| | Arabic / ...             |  | Chinese / ...             |     |
| | Size: 320 MB             |  | Size: 350 MB              |     |
| | Status: Not installed    |  | Status: Not installed     |     |
| |                          |  |                           |     |
| | [Download]               |  | [Download]                |     |
| +---------------------------+  +---------------------------+     |
|                                                                  |
| UNIVERSAL TRANSLATOR (Recommended for multi-language)            |
| +----------------------------------------------------------+    |
| | NLLB-200 Universal                                        |    |
| | Size: 2.4 GB                                              |    |
| | Languages: 200                                            |    |
| | RAM Required: 8GB+                                        |    |
| | Status: Not installed                                     |    |
| |                                                           |    |
| | Best for: Community centers, NGOs, multi-lingual areas    |    |
| |                                                           |    |
| | [Download]                                                |    |
| +----------------------------------------------------------+    |
|                                                                  |
| TRANSLATION CACHE                                                |
| ---------------------------------------------------------------- |
| Cache size: 45 MB / 500 MB limit                                 |
| Cached articles: 127                                             |
| [Clear Cache]                                                    |
|                                                                  |
+------------------------------------------------------------------+
```

---

## API Endpoints Needed

### List Available Languages

```
GET /api/languages
Response:
{
  "installed": ["en", "es"],
  "available": ["fr", "ar", "zh", "pt", "hi"],
  "active": "es",
  "nllb_installed": false
}
```

### Download Language Pack

```
POST /api/download-language
Body: { "language": "fr" }
Response:
{
  "job_id": "abc123",
  "status": "downloading",
  "progress": 0
}
```

### Set Active Language

```
POST /api/set-language
Body: { "language": "es" }
Response:
{
  "success": true,
  "language": "es"
}
```

### Get Translation Cache Status

```
GET /api/translation-cache
Response:
{
  "size_mb": 45,
  "max_mb": 500,
  "article_count": 127,
  "languages": ["es"]
}
```

### Clear Translation Cache

```
POST /api/clear-translation-cache
Body: { "language": "es" }  // optional, clears all if omitted
Response:
{
  "success": true,
  "cleared_mb": 45
}
```

---

## Dependencies

Required Python packages for translation:

```
transformers>=4.30.0
sentencepiece>=0.1.99
sacremoses>=0.0.53
beautifulsoup4>=4.12.0
```

For NLLB:
```
protobuf>=3.20.0
```

---

*Last updated: December 2025*
