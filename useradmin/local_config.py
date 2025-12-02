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
        "cache_responses": True,       # Cache LLM responses for offline use
        "last_sync": None,             # Last time sources were synced
        "ui_preferences": {
            "theme": "dark",
            "show_scores": True,
            "articles_per_page": 10
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config manager with optional custom path"""
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Default to config folder in project root
            self.config_path = Path(__file__).parent.parent / "config" / "local_settings.json"

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
            if file_type == "zim" and entry.is_file():
                # ZIM files are single files with .zim extension
                if entry.name.lower().endswith('.zim'):
                    stat = entry.stat()
                    files.append({
                        "name": entry.name,
                        "path": entry.path,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": "zim"
                    })

            elif file_type == "html" and entry.is_dir():
                # HTML backups are folders (like "appropedia", "builditsolar")
                # Skip special folders like "pdfs"
                if entry.name.lower() not in ['pdfs', 'pdf', 'zim', 'zims']:
                    # Check if it contains HTML files or looks like a site backup
                    dir_path = Path(entry.path)
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
                            "type": "html"
                        })

            elif file_type == "pdf":
                if entry.is_file() and entry.name.lower().endswith('.pdf'):
                    # PDF files in root
                    stat = entry.stat()
                    files.append({
                        "name": entry.name,
                        "path": entry.path,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": "pdf"
                    })
                elif entry.is_dir() and entry.name.lower() in ['pdfs', 'pdf']:
                    # Also check for pdfs subfolder
                    pdf_path = Path(entry.path)
                    for pdf_file in pdf_path.glob('*.pdf'):
                        stat = pdf_file.stat()
                        files.append({
                            "name": pdf_file.name,
                            "path": str(pdf_file),
                            "size_mb": round(stat.st_size / (1024 * 1024), 2),
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "type": "pdf"
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
