"""
Cloud Proxy Client
Allows local admins to access R2 cloud storage through Railway proxy endpoints.
Used when local admin doesn't have R2 keys configured.
"""

import requests
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from admin.local_config import get_local_config
from admin.connection_manager import get_connection_manager


class CloudProxyClient:
    """
    Client for accessing cloud resources through Railway proxy.

    Used when:
    - Local admin doesn't have R2 keys
    - Railway proxy URL is configured
    """

    def __init__(self, proxy_url: Optional[str] = None):
        if proxy_url:
            self.proxy_url = proxy_url.rstrip("/")
        else:
            config = get_local_config()
            self.proxy_url = config.get_railway_proxy_url()

        self.timeout = 30  # seconds
        self._last_error = None

    def is_configured(self) -> bool:
        """Check if proxy URL is configured"""
        return bool(self.proxy_url)

    def get_last_error(self) -> Optional[str]:
        """Get last error message"""
        return self._last_error

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle response and update connection manager status"""
        conn = get_connection_manager()

        if response.status_code >= 500:
            # Server error - mark global system as unavailable
            conn.on_global_system_error()
            self._last_error = f"Server error: {response.status_code}"
            return {"error": self._last_error, "status_code": response.status_code}

        if response.status_code >= 400:
            # Client error - not a system issue
            self._last_error = response.json().get("message", f"Error: {response.status_code}")
            return {"error": self._last_error, "status_code": response.status_code}

        # Success - mark global system as available
        conn.on_global_system_ok()
        self._last_error = None
        return response.json()

    def get_sources(self) -> Dict[str, Any]:
        """
        Get available sources from cloud.

        Returns:
            Dict with 'sources' list and 'connected' bool
        """
        if not self.is_configured():
            return {"error": "Proxy URL not configured", "sources": [], "connected": False}

        try:
            response = requests.get(
                f"{self.proxy_url}/api/cloud/sources",
                timeout=self.timeout
            )
            return self._handle_response(response)
        except requests.exceptions.Timeout:
            get_connection_manager().on_global_system_error()
            self._last_error = "Connection timeout"
            return {"error": self._last_error, "sources": [], "connected": False}
        except requests.exceptions.ConnectionError:
            get_connection_manager().on_global_system_error()
            self._last_error = "Connection failed"
            return {"error": self._last_error, "sources": [], "connected": False}
        except Exception as e:
            self._last_error = str(e)
            return {"error": self._last_error, "sources": [], "connected": False}

    def get_source_files(self, source_id: str) -> Dict[str, Any]:
        """
        Get list of files for a source.

        Returns:
            Dict with 'files' list, 'total_files', 'total_size_mb'
        """
        if not self.is_configured():
            return {"error": "Proxy URL not configured", "files": []}

        try:
            response = requests.get(
                f"{self.proxy_url}/api/cloud/download/{source_id}",
                timeout=self.timeout
            )
            return self._handle_response(response)
        except requests.exceptions.Timeout:
            get_connection_manager().on_global_system_error()
            self._last_error = "Connection timeout"
            return {"error": self._last_error, "files": []}
        except Exception as e:
            self._last_error = str(e)
            return {"error": self._last_error, "files": []}

    def download_file(self, source_id: str, filename: str, local_path: str) -> bool:
        """
        Download a file from cloud via proxy.

        Args:
            source_id: Source identifier (e.g., 'appropedia')
            filename: File name within the source
            local_path: Where to save locally

        Returns:
            True if successful
        """
        if not self.is_configured():
            self._last_error = "Proxy URL not configured"
            return False

        try:
            response = requests.get(
                f"{self.proxy_url}/api/cloud/download/{source_id}/{filename}",
                timeout=120,  # Longer timeout for downloads
                stream=True
            )

            if response.status_code != 200:
                get_connection_manager().on_global_system_error()
                self._last_error = f"Download failed: {response.status_code}"
                return False

            # Mark system as available
            get_connection_manager().on_global_system_ok()

            # Stream to file
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            self._last_error = None
            return True

        except requests.exceptions.Timeout:
            get_connection_manager().on_global_system_error()
            self._last_error = "Download timeout"
            return False
        except Exception as e:
            self._last_error = str(e)
            return False

    def chat(self, message: str, session_id: Optional[str] = None,
             sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send a chat request via Railway proxy.
        Used when local admin doesn't have Pinecone key but wants online search.

        Args:
            message: User's message/query
            session_id: Optional session ID for conversation continuity
            sources: Optional list of source IDs to filter

        Returns:
            Dict with 'response', 'session_id', or 'error'
        """
        if not self.is_configured():
            return {"error": "Proxy URL not configured", "response": ""}

        try:
            payload = {"message": message}
            if session_id:
                payload["session_id"] = session_id
            if sources is not None:
                payload["sources"] = sources

            response = requests.post(
                f"{self.proxy_url}/api/v1/chat",
                json=payload,
                timeout=60  # Longer timeout for LLM response
            )
            return self._handle_response(response)
        except requests.exceptions.Timeout:
            get_connection_manager().on_global_system_error()
            self._last_error = "Chat request timeout"
            return {"error": self._last_error, "response": ""}
        except requests.exceptions.ConnectionError:
            get_connection_manager().on_global_system_error()
            self._last_error = "Connection failed"
            return {"error": self._last_error, "response": ""}
        except Exception as e:
            self._last_error = str(e)
            return {"error": self._last_error, "response": ""}

    def search(self, query: str, n_results: int = 10,
               sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search via Railway proxy chat endpoint.
        Returns articles from the proxy's search results.

        This is a lighter-weight call than full chat - it extracts just
        the search results without needing the full LLM response.

        Args:
            query: Search query
            n_results: Number of results desired
            sources: Optional source filter

        Returns:
            Dict with 'articles' list or 'error'
        """
        if not self.is_configured():
            return {"error": "Proxy URL not configured", "articles": []}

        try:
            # Use the streaming endpoint to get articles without waiting for LLM
            payload = {"message": query}
            if sources is not None:
                payload["sources"] = sources

            response = requests.post(
                f"{self.proxy_url}/api/v1/chat/stream",
                json=payload,
                timeout=30,
                stream=True
            )

            if response.status_code != 200:
                get_connection_manager().on_global_system_error()
                self._last_error = f"Search failed: {response.status_code}"
                return {"error": self._last_error, "articles": []}

            # Parse SSE response to extract articles
            # Articles are sent first as [ARTICLES]<json>
            articles = []
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: [ARTICLES]"):
                    articles_json = line.replace("data: [ARTICLES]", "")
                    articles = json.loads(articles_json)
                    break

            get_connection_manager().on_global_system_ok()
            self._last_error = None
            return {"articles": articles[:n_results]}

        except requests.exceptions.Timeout:
            get_connection_manager().on_global_system_error()
            self._last_error = "Search timeout"
            return {"error": self._last_error, "articles": []}
        except Exception as e:
            self._last_error = str(e)
            return {"error": self._last_error, "articles": []}

    def submit_content(self, source_name: str, title: str, content: str,
                       url: Optional[str] = None,
                       submitter_name: Optional[str] = None,
                       submitter_email: Optional[str] = None,
                       notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Submit content for review.

        Returns:
            Dict with 'status', 'submission_id', 'message'
        """
        if not self.is_configured():
            return {"error": "Proxy URL not configured"}

        try:
            response = requests.post(
                f"{self.proxy_url}/api/cloud/submit",
                json={
                    "source_name": source_name,
                    "title": title,
                    "content": content,
                    "url": url,
                    "submitter_name": submitter_name,
                    "submitter_email": submitter_email,
                    "notes": notes
                },
                timeout=self.timeout
            )
            return self._handle_response(response)
        except Exception as e:
            self._last_error = str(e)
            return {"error": self._last_error}


# Singleton
_proxy_client: Optional[CloudProxyClient] = None


def get_proxy_client() -> CloudProxyClient:
    """Get or create proxy client singleton"""
    global _proxy_client
    if _proxy_client is None:
        _proxy_client = CloudProxyClient()
    return _proxy_client


def reset_proxy_client():
    """Reset proxy client (useful when config changes)"""
    global _proxy_client
    _proxy_client = None
