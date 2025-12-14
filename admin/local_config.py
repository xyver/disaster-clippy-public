"""
Local User Configuration Manager
Handles user-specific settings for their offline disaster preparedness system
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class BackupFolderNotConfigured(Exception):
    """
    Raised when backup folder is required but not configured.

    This provides a clear error message directing users to configure
    their backup folder in Settings or .env file.
    """
    pass


class LocalConfig:
    """Manages local user settings stored in a JSON file"""

    DEFAULT_CONFIG = {
        "backup_folder": "",           # Single unified backup folder path
        "backup_paths": {              # Legacy - kept for compatibility
            "zim_folder": "",
            "html_folder": "",
            "pdf_folder": ""
        },
        "offline_mode": "hybrid",      # "online_only", "hybrid", "offline_only"
        "auto_fallback": True,         # Automatically use offline when internet unavailable
        "selected_sources": [],        # Empty = all sources; otherwise list of source IDs
        "railway_proxy_url": "",       # URL of Railway deployment for cloud proxy (e.g., https://disaster-clippy.up.railway.app)
        "cache_responses": True,       # Cache LLM responses for offline use
        "last_sync": None,             # Last time sources were synced
        "ui_preferences": {
            "theme": "dark",
            "show_scores": True,
            "articles_per_page": 10
        },
        "ollama": {
            "enabled": False,          # Use Ollama for offline LLM
            "url": "http://localhost:11434",  # Ollama API URL
            "model": "mistral",        # Model to use
            "auto_start": True,        # Auto-start portable Ollama when in offline mode
            "portable_path": ""        # Path to portable Ollama (empty = use BACKUP_PATH/ollama)
        },
        "gpu": {
            "enabled": False,          # Enable GPU acceleration (requires CUDA)
            "llm_layers": -1,          # GPU layers for LLM (-1 = all, 0 = none, N = specific count)
            "auto_detect": True        # Auto-detect GPU and enable if available
        },
        "prompts": {
            # Unified prompt (used for all LLMs - format is auto-detected by llama_runtime)
            "system": """You are Disaster Clippy, a helpful assistant for DIY guides and humanitarian resources.

Help users find what they need through natural conversation. Recommend 3-5 relevant articles with brief descriptions.

Guidelines:
- Answer questions directly based on the article content provided
- ALWAYS format article titles as clickable markdown links: [Article Title](URL)
- Include 3-5 article recommendations in each response
- Keep responses concise but informative
- ONLY recommend articles from the provided context - never make up articles

Example format:
Here are some guides that can help:
1. [How to Build a Solar Panel](/zim/appropedia/wiki/Solar_Panel) - Step-by-step instructions for DIY solar
2. [Water Purification Methods](/zim/appropedia/wiki/Water) - Overview of filtration techniques"""
        },
        "personal_cloud": {
            "enabled": False,
            "provider": "r2",  # r2, s3, backblaze, digitalocean, custom
            "endpoint_url": "",
            "access_key_id": "",
            "secret_access_key": "",
            "bucket_name": "",
            "region": "auto"
        },
        "models": {
            "embedding_model": "all-mpnet-base-v2",  # Active embedding model
            "llm_model": None,                        # Active LLM model
            "llm_runtime": "llama.cpp"               # LLM runtime: llama.cpp or ollama
        },
        "translation": {
            "enabled": False,                         # Enable article translation
            "active_language": "en",                  # Target language code ("en" = no translation)
            "cache_enabled": True,                    # Cache translated articles
            "cache_max_mb": 500                       # Max cache size in MB
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config manager with optional custom path"""
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Default to root level local_settings.json
            self.config_path = Path(__file__).parent.parent / "local_settings.json"

        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load config from file or create default"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                # Merge with defaults to handle new fields
                return self._merge_config(self.DEFAULT_CONFIG.copy(), saved_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config: {e}. Using defaults.")
                return self.DEFAULT_CONFIG.copy()
        return self.DEFAULT_CONFIG.copy()

    def _merge_config(self, default: Dict, saved: Dict) -> Dict:
        """Recursively merge saved config with defaults"""
        result = default.copy()
        for key, value in saved.items():
            if key in result:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._merge_config(result[key], value)
                else:
                    result[key] = value
        return result

    def save(self) -> bool:
        """Save current config to file"""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, default=str)
            return True
        except IOError as e:
            print(f"Error saving config: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a config value by key (supports dot notation)"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    # Convenience methods for common operations

    def get_backup_folder(self) -> str:
        """Get the unified backup folder path"""
        # Try new unified path first, fall back to legacy zim_folder, then env var
        folder = self.config.get("backup_folder", "")
        if not folder:
            folder = self.config.get("backup_paths", {}).get("zim_folder", "")
        if not folder:
            folder = os.getenv("BACKUP_PATH", "")
        return folder

    def require_backup_folder(self) -> str:
        """
        Get backup folder or raise a clear error if not configured.

        Use this when backup folder is required for an operation.

        Returns:
            str: The backup folder path

        Raises:
            BackupFolderNotConfigured: If backup folder is not set
        """
        folder = self.get_backup_folder()
        if not folder:
            raise BackupFolderNotConfigured(
                "Backup folder not configured. "
                "Please set it in Settings page (http://localhost:8001/useradmin/settings) "
                "or set BACKUP_PATH in your .env file."
            )
        if not os.path.isdir(folder):
            raise BackupFolderNotConfigured(
                f"Backup folder does not exist: {folder}. "
                "Please create the folder or update the path in Settings."
            )
        return folder

    def set_backup_folder(self, path: str) -> None:
        """Set the unified backup folder path"""
        self.config["backup_folder"] = path
        # Also set legacy paths for compatibility
        if "backup_paths" not in self.config:
            self.config["backup_paths"] = {}
        self.config["backup_paths"]["zim_folder"] = path
        self.config["backup_paths"]["html_folder"] = path
        self.config["backup_paths"]["pdf_folder"] = path

        # Also update .env file so all code sees the same path
        self._update_env_backup_path(path)

    def _update_env_backup_path(self, path: str) -> None:
        """Update BACKUP_PATH in .env file"""
        env_path = Path(__file__).parent.parent / ".env"
        if not env_path.exists():
            return

        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Find and update BACKUP_PATH line
            found = False
            for i, line in enumerate(lines):
                if line.startswith("BACKUP_PATH="):
                    lines[i] = f"BACKUP_PATH={path}\n"
                    found = True
                    break

            # Add if not found
            if not found:
                lines.append(f"\nBACKUP_PATH={path}\n")

            with open(env_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            # Also update os.environ so current process sees it
            os.environ["BACKUP_PATH"] = path
        except Exception as e:
            print(f"Warning: Could not update .env file: {e}")

    def get_backup_paths(self) -> Dict[str, str]:
        """Get all backup folder paths (legacy compatibility)"""
        # Return unified folder for all paths
        folder = self.get_backup_folder()
        return {
            "zim_folder": folder,
            "html_folder": folder,
            "pdf_folder": folder,
            "backup_folder": folder
        }

    def set_backup_path(self, path_type: str, path: str) -> None:
        """Set a backup folder path - now sets unified folder"""
        self.set_backup_folder(path)

    def get_offline_mode(self) -> str:
        """Get current offline mode setting"""
        return self.config.get("offline_mode", "hybrid")

    def set_offline_mode(self, mode: str) -> None:
        """Set offline mode (online_only, hybrid, offline_only)"""
        if mode in ["online_only", "hybrid", "offline_only"]:
            self.config["offline_mode"] = mode

    def get_selected_sources(self) -> List[str]:
        """Get list of selected source IDs"""
        return self.config.get("selected_sources", [])

    def set_selected_sources(self, sources: List[str]) -> None:
        """Set selected sources"""
        self.config["selected_sources"] = sources

    def update_last_sync(self) -> None:
        """Update last sync timestamp"""
        self.config["last_sync"] = datetime.now().isoformat()

    # Railway Proxy settings (for local admins without R2 keys)
    def get_railway_proxy_url(self) -> str:
        """Get Railway proxy URL for cloud operations"""
        # First check environment variable, then config
        env_url = os.getenv("RAILWAY_PROXY_URL", "")
        if env_url:
            return env_url.rstrip("/")
        return self.config.get("railway_proxy_url", "").rstrip("/")

    def set_railway_proxy_url(self, url: str) -> None:
        """Set Railway proxy URL"""
        self.config["railway_proxy_url"] = url.rstrip("/") if url else ""

    def should_use_proxy(self) -> bool:
        """
        Check if we should use Railway proxy for cloud operations.
        Returns True if:
        - R2 keys are NOT configured locally
        - Railway proxy URL IS configured
        """
        from offline_tools.cloud.r2 import get_r2_storage
        storage = get_r2_storage()
        has_r2_keys = storage.is_configured()
        has_proxy_url = bool(self.get_railway_proxy_url())
        return not has_r2_keys and has_proxy_url

    def should_use_proxy_for_search(self) -> bool:
        """
        Check if we should use Railway proxy for online search.
        Returns True if:
        - Pinecone key is NOT configured locally (VECTOR_DB_MODE != pinecone or no key)
        - Railway proxy URL IS configured
        - We're in online or hybrid mode (not offline_only)

        This allows local admins without Pinecone keys to still use
        online semantic search through the Railway deployment.
        """
        # Check if we're in offline_only mode - no proxy needed
        if self.get_offline_mode() == "offline_only":
            return False

        # Check if proxy URL is configured
        proxy_url = self.get_railway_proxy_url()
        if not proxy_url:
            return False

        # Check if Pinecone is configured locally
        vector_mode = os.getenv("VECTOR_DB_MODE", "local").lower()
        pinecone_key = os.getenv("PINECONE_API_KEY", "")

        # If using pinecone mode with a key, don't need proxy
        if vector_mode == "pinecone" and pinecone_key:
            return False

        # No Pinecone configured, use proxy for online search
        return True

    # Ollama settings
    def get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama configuration"""
        return self.config.get("ollama", self.DEFAULT_CONFIG["ollama"])

    def set_ollama_config(self, **kwargs) -> None:
        """Update Ollama configuration with provided values"""
        if "ollama" not in self.config:
            self.config["ollama"] = self.DEFAULT_CONFIG["ollama"].copy()
        for key, value in kwargs.items():
            if key in self.config["ollama"]:
                self.config["ollama"][key] = value

    def is_ollama_enabled(self) -> bool:
        """Check if Ollama is enabled for offline LLM"""
        return self.config.get("ollama", {}).get("enabled", False)

    def get_ollama_url(self) -> str:
        """Get Ollama API URL"""
        return self.config.get("ollama", {}).get("url", "http://localhost:11434")

    def get_ollama_model(self) -> str:
        """Get Ollama model name"""
        return self.config.get("ollama", {}).get("model", "mistral")

    def get_ollama_portable_path(self) -> str:
        """Get path to portable Ollama installation"""
        custom_path = self.config.get("ollama", {}).get("portable_path", "")
        if custom_path:
            return custom_path
        # Default to BACKUP_PATH/ollama
        backup_folder = self.get_backup_folder()
        if backup_folder:
            return os.path.join(backup_folder, "ollama")
        return ""

    # Prompt settings (unified - same prompt for online and offline)
    def get_prompt(self, mode: str = "online") -> str:
        """Get system prompt for chat (mode param kept for backward compatibility)"""
        prompts = self.config.get("prompts", self.DEFAULT_CONFIG["prompts"])
        # Prefer unified "system" prompt, fall back to mode-specific for legacy configs
        return prompts.get("system", prompts.get(mode, prompts.get("online", "")))

    def set_prompt(self, mode: str = None, prompt: str = None) -> None:
        """Set unified system prompt (mode param ignored - kept for backward compatibility)"""
        if "prompts" not in self.config:
            self.config["prompts"] = self.DEFAULT_CONFIG["prompts"].copy()
        self.config["prompts"]["system"] = prompt

    def get_prompts(self) -> Dict[str, str]:
        """Get all prompts"""
        return self.config.get("prompts", self.DEFAULT_CONFIG["prompts"])

    def reset_prompt(self, mode: str = None) -> str:
        """Reset prompt to default and return the default value"""
        default_prompt = self.DEFAULT_CONFIG["prompts"].get("system", "")
        self.set_prompt(prompt=default_prompt)
        return default_prompt

    # Personal Cloud Backup settings
    def get_personal_cloud_config(self) -> Dict[str, Any]:
        """Get personal cloud backup configuration"""
        return self.config.get("personal_cloud", self.DEFAULT_CONFIG["personal_cloud"])

    def get_personal_cloud_config_for_ui(self) -> Dict[str, Any]:
        """Get cloud config with masked credentials for UI display"""
        cloud = self.get_personal_cloud_config().copy()
        if cloud.get("access_key_id"):
            key = cloud["access_key_id"]
            if len(key) > 4:
                cloud["access_key_id"] = "*" * (len(key) - 4) + key[-4:]
            else:
                cloud["access_key_id"] = "*" * len(key)
        if cloud.get("secret_access_key"):
            cloud["secret_access_key"] = "*" * 20
        return cloud

    def set_personal_cloud_config(self, **kwargs) -> None:
        """Update personal cloud configuration with provided values"""
        if "personal_cloud" not in self.config:
            self.config["personal_cloud"] = self.DEFAULT_CONFIG["personal_cloud"].copy()
        for key, value in kwargs.items():
            if key in self.config["personal_cloud"]:
                self.config["personal_cloud"][key] = value

    def is_personal_cloud_enabled(self) -> bool:
        """Check if personal cloud backup is enabled"""
        return self.config.get("personal_cloud", {}).get("enabled", False)

    # Models settings
    def get_models_config(self) -> Dict[str, Any]:
        """Get models configuration"""
        return self.config.get("models", self.DEFAULT_CONFIG["models"])

    def set_models_config(self, **kwargs) -> None:
        """Update models configuration with provided values"""
        if "models" not in self.config:
            self.config["models"] = self.DEFAULT_CONFIG["models"].copy()
        for key, value in kwargs.items():
            if key in self.config["models"]:
                self.config["models"][key] = value

    def get_embedding_model(self) -> Optional[str]:
        """Get the configured embedding model name"""
        return self.config.get("models", {}).get("embedding_model", "all-mpnet-base-v2")

    def set_embedding_model(self, model_name: str) -> None:
        """Set the active embedding model"""
        if "models" not in self.config:
            self.config["models"] = self.DEFAULT_CONFIG["models"].copy()
        self.config["models"]["embedding_model"] = model_name

    def get_llm_model(self) -> Optional[str]:
        """Get the configured LLM model name"""
        return self.config.get("models", {}).get("llm_model")

    def set_llm_model(self, model_name: Optional[str]) -> None:
        """Set the active LLM model"""
        if "models" not in self.config:
            self.config["models"] = self.DEFAULT_CONFIG["models"].copy()
        self.config["models"]["llm_model"] = model_name

    def get_llm_runtime(self) -> str:
        """Get the LLM runtime (llama.cpp or ollama)"""
        return self.config.get("models", {}).get("llm_runtime", "llama.cpp")

    def set_llm_runtime(self, runtime: str) -> None:
        """Set the LLM runtime"""
        if runtime not in ["llama.cpp", "ollama"]:
            raise ValueError("Runtime must be 'llama.cpp' or 'ollama'")
        if "models" not in self.config:
            self.config["models"] = self.DEFAULT_CONFIG["models"].copy()
        self.config["models"]["llm_runtime"] = runtime

    # Translation settings
    def get_translation_language(self) -> str:
        """Get the active translation language code"""
        return self.config.get("translation", {}).get("active_language", "en")

    def set_translation_language(self, lang_code: str) -> None:
        """Set the active translation language"""
        if "translation" not in self.config:
            self.config["translation"] = self.DEFAULT_CONFIG["translation"].copy()
        self.config["translation"]["active_language"] = lang_code
        # Enable translation if setting a non-English language
        self.config["translation"]["enabled"] = (lang_code != "en")

    def is_translation_enabled(self) -> bool:
        """Check if translation is enabled"""
        return self.config.get("translation", {}).get("enabled", False)

    def set_translation_enabled(self, enabled: bool) -> None:
        """Enable or disable translation"""
        if "translation" not in self.config:
            self.config["translation"] = self.DEFAULT_CONFIG["translation"].copy()
        self.config["translation"]["enabled"] = enabled

    def is_translation_cache_enabled(self) -> bool:
        """Check if translation caching is enabled"""
        return self.config.get("translation", {}).get("cache_enabled", True)

    def get_translation_cache_max_mb(self) -> int:
        """Get max translation cache size in MB"""
        return self.config.get("translation", {}).get("cache_max_mb", 500)

    # GPU Configuration
    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is enabled"""
        gpu_config = self.config.get("gpu", {})
        if gpu_config.get("auto_detect", True):
            return self._detect_gpu_available()
        return gpu_config.get("enabled", False)

    def get_gpu_llm_layers(self) -> int:
        """
        Get number of LLM layers to offload to GPU.
        Returns: -1 for all layers, 0 for none, or specific count
        """
        if not self.is_gpu_enabled():
            return 0
        return self.config.get("gpu", {}).get("llm_layers", -1)

    def set_gpu_enabled(self, enabled: bool) -> None:
        """Enable or disable GPU acceleration"""
        if "gpu" not in self.config:
            self.config["gpu"] = self.DEFAULT_CONFIG["gpu"].copy()
        self.config["gpu"]["enabled"] = enabled
        self.config["gpu"]["auto_detect"] = False  # Manual override

    def set_gpu_llm_layers(self, layers: int) -> None:
        """Set number of LLM layers to offload to GPU"""
        if "gpu" not in self.config:
            self.config["gpu"] = self.DEFAULT_CONFIG["gpu"].copy()
        self.config["gpu"]["llm_layers"] = layers

    def _detect_gpu_available(self) -> bool:
        """Auto-detect if CUDA GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        # Also check for llama-cpp-python CUDA support
        try:
            from llama_cpp import Llama
            # If llama_cpp is built with CUDA, this should work
            return True  # Assume available if llama_cpp is installed
        except ImportError:
            pass
        return False


def scan_backup_folder(folder_path: str, file_type: str = "zim") -> List[Dict[str, Any]]:
    """
    Scan a folder for backup files and return info about each

    Args:
        folder_path: Path to scan
        file_type: Type of files to look for ("zim", "html", "pdf")

    Returns:
        List of dicts with file info (name, path, size, modified)
    """
    if not folder_path or not os.path.isdir(folder_path):
        return []

    files = []

    try:
        for entry in os.scandir(folder_path):
            if file_type == "zim":
                # ZIM files can be in root (legacy) or inside source folders (new)
                if entry.is_file() and entry.name.lower().endswith('.zim'):
                    # Legacy: ZIM in root folder
                    stat = entry.stat()
                    files.append({
                        "name": entry.name,
                        "path": entry.path,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": "zim"
                    })
                elif entry.is_dir():
                    # New: check inside source folders for ZIM files
                    dir_path = Path(entry.path)
                    for zim_file in dir_path.glob("*.zim"):
                        # Calculate FULL folder size (ZIM + metadata files)
                        try:
                            total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                            size_mb = round(total_size / (1024 * 1024), 2)
                        except (PermissionError, OSError):
                            size_mb = round(zim_file.stat().st_size / (1024 * 1024), 2)

                        stat = zim_file.stat()
                        files.append({
                            "name": zim_file.name,
                            "path": str(zim_file),
                            "size_mb": size_mb,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "type": "zim",
                            "source_id": entry.name  # The folder name is the source_id
                        })

            elif file_type == "html" and entry.is_dir():
                # HTML backups are folders (like "appropedia", "builditsolar")
                # Skip special folders like "pdfs"
                if entry.name.lower() not in ['pdfs', 'pdf', 'zim', 'zims']:
                    dir_path = Path(entry.path)

                    # Skip if this is a PDF collection (has _collection.json)
                    if (dir_path / "_collection.json").exists():
                        continue

                    # Skip if this is a ZIM source (has .zim file)
                    if list(dir_path.glob("*.zim")):
                        continue

                    # Check if it contains HTML files or looks like a site backup
                    has_html = any(dir_path.rglob('*.html')) or any(dir_path.rglob('*.htm'))
                    has_index = (dir_path / 'index.html').exists() or (dir_path / 'index.htm').exists()

                    if has_html or has_index:
                        # Calculate folder size
                        try:
                            total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                            size_mb = round(total_size / (1024 * 1024), 2)
                        except (PermissionError, OSError):
                            size_mb = 0

                        stat = entry.stat()
                        files.append({
                            "name": entry.name,
                            "path": entry.path,
                            "size_mb": size_mb,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "type": "html",
                            "source_id": entry.name
                        })

            elif file_type == "pdf":
                # PDF collections are folders with _collection.json
                if entry.is_dir():
                    dir_path = Path(entry.path)
                    collection_file = dir_path / "_collection.json"
                    if collection_file.exists():
                        # This is a PDF collection source
                        try:
                            total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                            size_mb = round(total_size / (1024 * 1024), 2)
                        except (PermissionError, OSError):
                            size_mb = 0

                        stat = entry.stat()
                        files.append({
                            "name": entry.name,
                            "path": entry.path,
                            "size_mb": size_mb,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "type": "pdf",
                            "source_id": entry.name
                        })

    except PermissionError:
        pass

    return sorted(files, key=lambda x: x["name"])


def scan_all_backups(folder_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Scan a folder for all backup types at once.

    Args:
        folder_path: Path to scan

    Returns:
        Dict with keys: zim_files, html_folders, pdf_files
    """
    return {
        "zim_files": scan_backup_folder(folder_path, "zim"),
        "html_folders": scan_backup_folder(folder_path, "html"),
        "pdf_files": scan_backup_folder(folder_path, "pdf")
    }


_internet_check_cache = {"result": None, "timestamp": 0}

def check_internet_available() -> bool:
    """Quick check if internet is available (cached for 30 seconds)"""
    import socket
    import time

    # Return cached result if less than 30 seconds old
    cache_ttl = 30
    now = time.time()
    if _internet_check_cache["result"] is not None:
        if now - _internet_check_cache["timestamp"] < cache_ttl:
            return _internet_check_cache["result"]

    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        result = True
    except OSError:
        result = False

    _internet_check_cache["result"] = result
    _internet_check_cache["timestamp"] = now
    return result


# Singleton instance for easy access
_config_instance = None

def get_local_config() -> LocalConfig:
    """Get or create the singleton config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = LocalConfig()
    return _config_instance
