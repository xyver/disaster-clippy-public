"""
Local User Configuration Manager
Handles user-specific settings for their offline disaster preparedness system
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


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
        "prompts": {
            # Online mode prompt (used with cloud LLM like Claude/OpenAI)
            "online": """You are Disaster Clippy, a helpful assistant that helps people find DIY guides and humanitarian resources.

Your role is to:
1. Understand what the user needs help with
2. Suggest relevant articles from the knowledge base
3. Help them refine their search through conversation
4. Answer follow-up questions about the articles

When presenting search results:
- Summarize what each article is about in 1-2 sentences
- Explain why it might be relevant to their situation
- Offer to find more similar articles or narrow down the search

Be conversational, helpful, and practical. Focus on actionable solutions.

IMPORTANT: You can ONLY recommend articles that are provided to you in the context. Do not make up or hallucinate articles that don't exist in the search results.""",

            # Offline mode prompt (used with local Ollama - shorter for efficiency)
            "offline": """You are Disaster Clippy, a helpful assistant for DIY and humanitarian resources.
Your role is to help users find relevant articles and answer questions based on the provided context.
Be concise, practical, and helpful. Focus on actionable information.
Only recommend articles that are in the provided context - do not make up articles."""
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
        # Try new unified path first, fall back to legacy zim_folder
        folder = self.config.get("backup_folder", "")
        if not folder:
            folder = self.config.get("backup_paths", {}).get("zim_folder", "")
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

    # Prompt settings
    def get_prompt(self, mode: str = "online") -> str:
        """Get system prompt for chat (online or offline mode)"""
        prompts = self.config.get("prompts", self.DEFAULT_CONFIG["prompts"])
        return prompts.get(mode, prompts.get("online", ""))

    def set_prompt(self, mode: str, prompt: str) -> None:
        """Set system prompt for a mode (online or offline)"""
        if "prompts" not in self.config:
            self.config["prompts"] = self.DEFAULT_CONFIG["prompts"].copy()
        self.config["prompts"][mode] = prompt

    def get_prompts(self) -> Dict[str, str]:
        """Get all prompts"""
        return self.config.get("prompts", self.DEFAULT_CONFIG["prompts"])

    def reset_prompt(self, mode: str) -> str:
        """Reset a prompt to default and return the default value"""
        default_prompt = self.DEFAULT_CONFIG["prompts"].get(mode, "")
        self.set_prompt(mode, default_prompt)
        return default_prompt


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


def check_internet_available() -> bool:
    """Quick check if internet is available"""
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


# Singleton instance for easy access
_config_instance = None

def get_local_config() -> LocalConfig:
    """Get or create the singleton config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = LocalConfig()
    return _config_instance
