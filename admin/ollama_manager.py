"""
Ollama Manager - Handles local LLM via Ollama

Supports both:
- User's existing Ollama installation
- Portable Ollama from BACKUP_PATH/ollama/
"""

import os
import json
import subprocess
import time
import platform
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Generator


class OllamaManager:
    """Manages Ollama process and API calls"""

    def __init__(self, url: str = "http://localhost:11434",
                 model: str = "mistral",
                 portable_path: str = None):
        self.url = url.rstrip("/")
        self.model = model
        self.portable_path = portable_path
        self._process: Optional[subprocess.Popen] = None

    def get_executable_path(self) -> Optional[str]:
        """Get path to Ollama executable"""
        if not self.portable_path:
            return None

        portable_dir = Path(self.portable_path)
        if not portable_dir.exists():
            return None

        # Check for platform-specific executable
        system = platform.system().lower()
        if system == "windows":
            exe_path = portable_dir / "ollama.exe"
        elif system == "darwin":
            exe_path = portable_dir / "ollama"
        else:  # Linux
            exe_path = portable_dir / "ollama"

        if exe_path.exists():
            return str(exe_path)
        return None

    def is_installed(self) -> bool:
        """Check if portable Ollama is installed"""
        return self.get_executable_path() is not None

    def is_running(self) -> bool:
        """Check if Ollama server is responding"""
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get detailed Ollama status"""
        status = {
            "installed": self.is_installed(),
            "running": False,
            "url": self.url,
            "model": self.model,
            "models_available": [],
            "portable_path": self.portable_path,
            "error": None
        }

        try:
            response = requests.get(f"{self.url}/api/tags", timeout=2)
            if response.status_code == 200:
                status["running"] = True
                data = response.json()
                status["models_available"] = [
                    m.get("name", "") for m in data.get("models", [])
                ]
        except requests.exceptions.ConnectionError:
            status["error"] = "Ollama not running"
        except Exception as e:
            status["error"] = str(e)

        return status

    def start(self) -> bool:
        """Start portable Ollama server"""
        exe_path = self.get_executable_path()
        if not exe_path:
            print("Portable Ollama not found")
            return False

        if self.is_running():
            print("Ollama already running")
            return True

        try:
            # Set models directory to be inside portable folder
            env = os.environ.copy()
            models_dir = Path(self.portable_path) / "models"
            env["OLLAMA_MODELS"] = str(models_dir)

            # Start Ollama serve in background
            if platform.system().lower() == "windows":
                # On Windows, use CREATE_NO_WINDOW to hide console
                self._process = subprocess.Popen(
                    [exe_path, "serve"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                self._process = subprocess.Popen(
                    [exe_path, "serve"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            # Wait for server to start
            for _ in range(10):  # Wait up to 5 seconds
                time.sleep(0.5)
                if self.is_running():
                    print(f"Ollama started successfully on {self.url}")
                    return True

            print("Ollama started but not responding")
            return False

        except Exception as e:
            print(f"Failed to start Ollama: {e}")
            return False

    def stop(self) -> bool:
        """Stop Ollama server if we started it"""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
                self._process = None
                print("Ollama stopped")
                return True
            except Exception as e:
                print(f"Error stopping Ollama: {e}")
                return False
        return True

    def generate(self, prompt: str, system: str = None,
                 context: str = None) -> Optional[str]:
        """
        Generate a response from Ollama.

        Args:
            prompt: The user's question/message
            system: System prompt (optional)
            context: Additional context (articles, etc.)

        Returns:
            Generated text or None on error
        """
        if not self.is_running():
            # Try to start if we have portable version
            if self.is_installed():
                if not self.start():
                    return None
            else:
                return None

        # Build the full prompt
        full_prompt = ""
        if system:
            full_prompt = f"{system}\n\n"
        if context:
            full_prompt += f"Context:\n{context}\n\n"
        full_prompt += f"User: {prompt}\n\nAssistant:"

        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 1024
                    }
                },
                timeout=60  # LLM can be slow
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                print(f"Ollama API error: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            print("Ollama request timed out")
            return None
        except Exception as e:
            print(f"Ollama generate error: {e}")
            return None

    def chat(self, messages: list, system: str = None) -> Optional[str]:
        """
        Chat with Ollama using message history.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system: System prompt

        Returns:
            Generated response or None on error
        """
        if not self.is_running():
            if self.is_installed():
                if not self.start():
                    return None
            else:
                return None

        try:
            # Build messages list for Ollama chat API
            ollama_messages = []
            if system:
                ollama_messages.append({"role": "system", "content": system})
            ollama_messages.extend(messages)

            response = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 1024
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("message", {}).get("content", "")
            else:
                print(f"Ollama chat API error: {response.status_code}")
                return None

        except Exception as e:
            print(f"Ollama chat error: {e}")
            return None

    def chat_stream(self, messages: list, system: str = None) -> Generator[str, None, None]:
        """
        Stream chat responses from Ollama.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system: System prompt

        Yields:
            Text chunks as they are generated
        """
        if not self.is_running():
            if self.is_installed():
                if not self.start():
                    return
            else:
                return

        try:
            # Build messages list for Ollama chat API
            ollama_messages = []
            if system:
                ollama_messages.append({"role": "system", "content": system})
            ollama_messages.extend(messages)

            response = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": ollama_messages,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 1024
                    }
                },
                timeout=120,
                stream=True
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                            # Check if done
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                print(f"Ollama chat stream API error: {response.status_code}")

        except Exception as e:
            print(f"Ollama chat stream error: {e}")


# Singleton instance
_ollama_manager: Optional[OllamaManager] = None


def get_ollama_manager() -> OllamaManager:
    """Get or create the Ollama manager singleton"""
    global _ollama_manager
    if _ollama_manager is None:
        from .local_config import get_local_config
        config = get_local_config()
        _ollama_manager = OllamaManager(
            url=config.get_ollama_url(),
            model=config.get_ollama_model(),
            portable_path=config.get_ollama_portable_path()
        )
    return _ollama_manager


def reload_ollama_manager() -> OllamaManager:
    """Reload Ollama manager with current config"""
    global _ollama_manager
    _ollama_manager = None
    return get_ollama_manager()
