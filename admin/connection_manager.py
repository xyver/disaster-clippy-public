"""
Connection Manager - Smart connectivity detection and mode switching

Handles:
- Connection mode (online_only, hybrid, offline_only)
- Smart ping with configurable intervals
- Automatic fallback detection in hybrid mode
- Status tracking for frontend display
- Connection state notifications (stable, unstable, checking, disconnected)
"""

import time
import asyncio
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from enum import Enum
import threading


class ConnectionState(Enum):
    """Connection states for UI display"""
    ONLINE = "online"              # Securely connected, recent successful call
    CHECKING = "checking"          # Currently verifying connection
    UNSTABLE = "unstable"          # In hybrid mode, experiencing intermittent issues
    DISCONNECTED = "disconnected"  # Connection lost (in online mode)
    OFFLINE = "offline"            # User chose offline mode (intentional)
    RECOVERING = "recovering"      # Was offline, now detecting recovery


class ConnectionManager:
    """
    Manages connection status and smart ping logic.

    Modes:
    - online_only: Always tries cloud APIs, shows error on failure
    - hybrid: Tries cloud first, falls back to local, keeps pinging to detect recovery
    - offline_only: Never tries cloud APIs, no pinging (user knows they're offline)

    States (for UI):
    - online: Stable connection, recent successful API call
    - checking: Currently pinging to verify connection
    - unstable: Hybrid mode with recent failures (using fallback)
    - disconnected: Online mode but connection lost
    - offline: User intentionally in offline mode
    - recovering: Was offline/unstable, now detecting recovery
    """

    # Default ping interval (5 minutes)
    DEFAULT_PING_INTERVAL = 5 * 60  # seconds

    # Minimum time between pings to avoid hammering
    MIN_PING_INTERVAL = 10  # seconds

    # Time since last success to consider "unstable"
    UNSTABLE_THRESHOLD = 60  # seconds

    # Number of recent failures to track
    FAILURE_HISTORY_SIZE = 5

    def __init__(self):
        self._is_online = True  # Assume online until proven otherwise
        self._last_ping_time = 0
        self._last_successful_api_call = 0
        self._ping_interval = self.DEFAULT_PING_INTERVAL
        self._mode = "hybrid"  # Default mode
        self._ping_lock = threading.Lock()
        self._status_callbacks = []  # Callbacks for status changes
        self._temporarily_offline = False  # In hybrid mode, tracks temporary outage
        self._is_checking = False  # Currently performing ping check
        self._recent_failures = []  # Track recent failure timestamps
        self._connection_state = ConnectionState.ONLINE

    def set_mode(self, mode: str):
        """Set connection mode"""
        if mode in ("online_only", "hybrid", "offline_only"):
            old_mode = self._mode
            self._mode = mode
            # Reset temporary offline when switching modes
            self._temporarily_offline = False
            self._recent_failures = []
            # Update state based on new mode
            self._update_connection_state()

    def get_mode(self) -> str:
        """Get current connection mode"""
        return self._mode

    def get_state(self) -> ConnectionState:
        """Get current connection state for UI display"""
        return self._connection_state

    def is_online(self) -> bool:
        """Check if currently online (or assumed online)"""
        if self._mode == "offline_only":
            return False  # User explicitly chose offline
        return self._is_online and not self._temporarily_offline

    def should_try_online(self) -> bool:
        """
        Determine if we should attempt online API calls.

        Returns:
            True if online APIs should be tried
        """
        if self._mode == "offline_only":
            return False  # Never try online in offline mode

        if self._mode == "online_only":
            return True  # Always try in online mode

        # Hybrid mode - try online unless we're in a known outage
        return not self._temporarily_offline

    def on_api_success(self):
        """
        Called when an API call succeeds.
        Resets the ping timer and marks as online.
        """
        was_state = self._connection_state
        self._is_online = True
        self._temporarily_offline = False
        self._last_successful_api_call = time.time()
        self._last_ping_time = time.time()  # Reset ping timer
        self._recent_failures = []  # Clear failure history on success
        self._update_connection_state()
        if was_state != self._connection_state:
            self._notify_status_change(True)

    def on_api_failure(self, error: Optional[Exception] = None):
        """
        Called when an API call fails.
        Tracks failures and triggers immediate ping check in hybrid/online modes.
        """
        if self._mode == "offline_only":
            return  # Don't care about failures in offline mode

        # Track failure
        now = time.time()
        self._recent_failures.append(now)
        # Keep only recent failures
        cutoff = now - 300  # Last 5 minutes
        self._recent_failures = [t for t in self._recent_failures if t > cutoff]

        # Update state
        self._update_connection_state()

        # Trigger immediate check
        self._check_connectivity()

    def _update_connection_state(self):
        """Update the connection state based on current conditions"""
        now = time.time()

        if self._mode == "offline_only":
            self._connection_state = ConnectionState.OFFLINE
            return

        if self._is_checking:
            self._connection_state = ConnectionState.CHECKING
            return

        if not self._is_online:
            if self._mode == "online_only":
                self._connection_state = ConnectionState.DISCONNECTED
            else:  # hybrid
                self._connection_state = ConnectionState.UNSTABLE
            return

        # Check for unstable (hybrid mode with recent failures)
        if self._mode == "hybrid" and len(self._recent_failures) >= 2:
            self._connection_state = ConnectionState.UNSTABLE
            return

        # Check if recovering (was offline, now coming back)
        if self._temporarily_offline and self._is_online:
            self._connection_state = ConnectionState.RECOVERING
            return

        # Check for stale connection (no recent success)
        time_since_success = now - self._last_successful_api_call
        if self._last_successful_api_call > 0 and time_since_success > self.UNSTABLE_THRESHOLD:
            if self._mode == "hybrid":
                self._connection_state = ConnectionState.UNSTABLE
            else:
                self._connection_state = ConnectionState.ONLINE  # Assume OK in online mode
            return

        # Stable online connection
        self._connection_state = ConnectionState.ONLINE

    def _check_connectivity(self) -> bool:
        """
        Perform actual connectivity check.
        Uses a lightweight ping to verify cloud services are reachable.
        """
        with self._ping_lock:
            # Avoid hammering
            now = time.time()
            if now - self._last_ping_time < self.MIN_PING_INTERVAL:
                return self._is_online

            self._last_ping_time = now
            self._is_checking = True
            self._update_connection_state()

        # Try to ping our health endpoint or a lightweight external service
        was_online = self._is_online
        was_state = self._connection_state

        try:
            import requests
            # Quick connectivity check - try to reach OpenAI API
            # Just checking if we can connect, not making a real API call
            response = requests.get(
                "https://api.openai.com/v1/models",
                timeout=3,
                headers={"Authorization": "Bearer test"}  # Will fail auth but proves connectivity
            )
            # Even a 401 means we reached the server
            self._is_online = response.status_code in (200, 401, 403)

        except Exception:
            # Any error, assume offline
            self._is_online = False
        finally:
            self._is_checking = False

        # Update temporary offline status for hybrid mode
        if self._mode == "hybrid":
            self._temporarily_offline = not self._is_online

        # Update state
        self._update_connection_state()

        # Notify if status changed
        if was_state != self._connection_state:
            self._notify_status_change(self._is_online)

        return self._is_online

    def should_ping(self) -> bool:
        """
        Check if it's time for a scheduled ping.

        Returns:
            True if a ping should be performed
        """
        if self._mode == "offline_only":
            return False  # No pinging in offline mode

        now = time.time()
        return (now - self._last_ping_time) >= self._ping_interval

    def perform_scheduled_ping(self) -> bool:
        """
        Perform a scheduled ping if needed.
        Call this periodically from a background task.

        Returns:
            Current online status
        """
        if self.should_ping():
            return self._check_connectivity()
        return self._is_online

    def get_status(self) -> Dict[str, Any]:
        """
        Get full connection status for frontend display.

        Returns dict with:
        - mode: User's selected mode (online_only, hybrid, offline_only)
        - state: Current connection state for UI display
        - state_label: Human-readable state label
        - state_color: Suggested color for UI (green, yellow, orange, red, gray)
        - is_online: Whether currently connected
        - message: Optional status message for user
        """
        state = self._connection_state
        state_info = self._get_state_info(state)

        return {
            "mode": self._mode,
            "state": state.value,
            "state_label": state_info["label"],
            "state_color": state_info["color"],
            "state_icon": state_info["icon"],
            "is_online": self._is_online,
            "temporarily_offline": self._temporarily_offline,
            "effective_mode": self._get_effective_mode(),
            "message": state_info["message"],
            "last_successful_call": self._last_successful_api_call,
            "last_ping": self._last_ping_time,
            "ping_interval": self._ping_interval,
            "recent_failures": len(self._recent_failures)
        }

    def _get_state_info(self, state: ConnectionState) -> Dict[str, str]:
        """Get display info for a connection state"""
        state_info = {
            ConnectionState.ONLINE: {
                "label": "Online",
                "color": "green",
                "icon": "check",
                "message": "Connected to cloud services"
            },
            ConnectionState.CHECKING: {
                "label": "Checking...",
                "color": "blue",
                "icon": "sync",
                "message": "Verifying connection..."
            },
            ConnectionState.UNSTABLE: {
                "label": "Unstable",
                "color": "yellow",
                "icon": "warning",
                "message": "Connection unstable - using fallback when needed"
            },
            ConnectionState.DISCONNECTED: {
                "label": "Disconnected",
                "color": "red",
                "icon": "error",
                "message": "Cannot reach cloud services"
            },
            ConnectionState.OFFLINE: {
                "label": "Offline",
                "color": "gray",
                "icon": "offline",
                "message": "Running in offline mode (by choice)"
            },
            ConnectionState.RECOVERING: {
                "label": "Reconnecting",
                "color": "blue",
                "icon": "sync",
                "message": "Connection restored - verifying..."
            }
        }
        return state_info.get(state, state_info[ConnectionState.ONLINE])

    def _get_effective_mode(self) -> str:
        """
        Get the effective operating mode (what's actually being used).
        Useful for hybrid mode where we might be temporarily offline.
        """
        if self._mode == "offline_only":
            return "offline"
        if self._mode == "online_only":
            return "online" if self._is_online else "online_disconnected"
        # Hybrid
        if self._temporarily_offline:
            return "hybrid_offline"
        return "hybrid_online"

    def register_status_callback(self, callback: Callable[[bool], None]):
        """Register a callback for status changes"""
        self._status_callbacks.append(callback)

    def _notify_status_change(self, is_online: bool):
        """Notify all registered callbacks of status change"""
        for callback in self._status_callbacks:
            try:
                callback(is_online)
            except Exception:
                pass  # Don't let callback errors break us


# Singleton instance
_connection_manager = None


def get_connection_manager() -> ConnectionManager:
    """Get or create the singleton connection manager"""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
        # Sync mode from local config if available
        try:
            from admin.local_config import get_local_config
            config = get_local_config()
            mode = config.get_offline_mode()
            _connection_manager.set_mode(mode)
        except Exception:
            pass
    return _connection_manager


def sync_mode_from_config():
    """Sync connection manager mode from local config"""
    try:
        from admin.local_config import get_local_config
        config = get_local_config()
        mode = config.get_offline_mode()
        get_connection_manager().set_mode(mode)
    except Exception:
        pass
