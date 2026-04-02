"""
App path utilities.

Central location for all path resolution. Separates install location (read-only
code) from data location (user-owned, writable).

All code that needs to write config, logs, or cache should use these functions
rather than deriving paths from __file__.
"""

import os
import sys
from pathlib import Path


def get_app_data_dir() -> Path:
    """
    Get the user-owned app data directory.

    This is where config, logs, and cache live. Source data (packs, vectors,
    models) live in BACKUP_PATH which may be different and is user-configurable.

    Priority:
    1. CLIPPY_DATA_DIR env var (explicit override, useful for testing and custom installs)
    2. Project root containing .env (dev workflow and portable native install)
       paths.py lives at admin/paths.py, so project root is two levels up.
       If a .env file is present there, config co-locates with it -- no %APPDATA% scatter.
    3. Platform default (fallback when no .env is found)

    Platform defaults:
      Windows   : %APPDATA%/DisasterClippy/
      macOS     : ~/Library/Application Support/DisasterClippy/
      Linux/Pi  : ~/.local/share/disaster-clippy/  (respects XDG_DATA_HOME)
    """
    override = os.getenv("CLIPPY_DATA_DIR", "")
    if override:
        return Path(override)

    project_root = Path(__file__).parent.parent
    if (project_root / ".env").exists():
        return project_root

    if sys.platform == "win32":
        base = os.getenv("APPDATA", "")
        if base:
            return Path(base) / "DisasterClippy"
        return Path.home() / "AppData" / "Roaming" / "DisasterClippy"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "DisasterClippy"

    # Linux, Raspberry Pi, and other Unix
    xdg = os.getenv("XDG_DATA_HOME", "")
    if xdg:
        return Path(xdg) / "disaster-clippy"
    return Path.home() / ".local" / "share" / "disaster-clippy"


def get_config_dir() -> Path:
    """Config subdirectory: settings.json and runtime.env live here."""
    return get_app_data_dir() / "config"


def get_log_dir() -> Path:
    """Logs subdirectory."""
    return get_app_data_dir() / "logs"


def get_cache_dir() -> Path:
    """Cache subdirectory: temporary exports, index exports, cached manifests."""
    return get_app_data_dir() / "cache"


def get_config_path() -> Path:
    """Full path to the main settings file."""
    return get_config_dir() / "settings.json"


def get_runtime_env_path() -> Path:
    """Full path to the runtime env file (installed runtime equivalent of .env)."""
    return get_config_dir() / "runtime.env"
