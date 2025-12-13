"""
llama.cpp Runtime for Offline LLM

Provides a simple interface for running local LLM models using llama-cpp-python.
This enables offline AI conversation without requiring an external service like Ollama.

Usage:
    from offline_tools.llama_runtime import get_llama_runtime

    runtime = get_llama_runtime()
    if runtime.is_available():
        response = runtime.chat([
            {"role": "user", "content": "How do I purify water?"}
        ])
        print(response)
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
import json


class LlamaRuntime:
    """
    Runtime for local LLM inference using llama-cpp-python.

    Loads GGUF model files from BACKUP_PATH/models/llm/ and provides
    chat/generate methods compatible with the Ollama interface.
    """

    def __init__(self, model_path: Optional[str] = None, n_ctx: int = 4096, n_gpu_layers: int = 0):
        """
        Initialize the llama runtime.

        Args:
            model_path: Path to GGUF model file. If None, auto-detects from model registry.
            n_ctx: Context window size (default 4096)
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
        """
        self._llm = None
        self._model_path = model_path
        self._model_name = None
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._load_error = None

        # Try to load model
        if model_path:
            self._load_model(model_path)
        else:
            self._auto_load_model()

    def _get_backup_folder(self) -> Optional[str]:
        """Get backup folder from config"""
        try:
            from admin.local_config import get_local_config
            return get_local_config().get_backup_folder()
        except Exception:
            return os.getenv("BACKUP_PATH", "")

    def _get_gpu_layers_from_config(self) -> int:
        """
        Get GPU layers from config with auto-detection.

        Returns:
            -1 if GPU available (all layers), 0 if CPU-only
        """
        try:
            from admin.local_config import get_local_config
            config = get_local_config()
            layers = config.get_gpu_llm_layers()
            if layers != 0:
                print(f"[LlamaRuntime] GPU acceleration enabled, using {layers} layers (-1 = all)")
            return layers
        except Exception as e:
            print(f"[LlamaRuntime] Could not read GPU config: {e}")
            return 0

    def _auto_load_model(self):
        """Auto-detect and load the configured LLM model"""
        try:
            from offline_tools.model_registry import get_model_registry, AVAILABLE_MODELS

            # Auto-detect GPU settings from config
            self._n_gpu_layers = self._get_gpu_layers_from_config()

            registry = get_model_registry()
            model_id = registry.get_installed_llm_model()

            if not model_id:
                self._load_error = "No LLM model installed"
                return

            model_info = AVAILABLE_MODELS.get(model_id)
            if not model_info:
                self._load_error = f"Unknown model: {model_id}"
                return

            model_path = registry.get_model_path(model_id)
            if not model_path:
                self._load_error = f"Model path not found: {model_id}"
                return

            filename = model_info.get("filename", "")
            full_path = model_path / filename

            if not full_path.exists():
                self._load_error = f"Model file not found: {full_path}"
                return

            self._model_name = model_id
            self._load_model(str(full_path))

        except Exception as e:
            self._load_error = f"Auto-load failed: {str(e)}"

    def _load_model(self, model_path: str):
        """Load a GGUF model file"""
        try:
            from llama_cpp import Llama

            print(f"[LlamaRuntime] Loading model: {model_path}")
            self._model_path = model_path

            self._llm = Llama(
                model_path=model_path,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False
            )

            print(f"[LlamaRuntime] Model loaded successfully")
            self._load_error = None

        except ImportError:
            self._load_error = "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            print(f"[LlamaRuntime] {self._load_error}")
        except Exception as e:
            self._load_error = f"Failed to load model: {str(e)}"
            print(f"[LlamaRuntime] {self._load_error}")

    def is_available(self) -> bool:
        """Check if the runtime is available and a model is loaded"""
        return self._llm is not None

    def get_status(self) -> Dict[str, Any]:
        """Get current runtime status"""
        return {
            "available": self.is_available(),
            "model_path": self._model_path,
            "model_name": self._model_name,
            "context_size": self._n_ctx,
            "gpu_layers": self._n_gpu_layers,
            "error": self._load_error
        }

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7,
                 stop: Optional[List[str]] = None) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            stop: List of stop sequences

        Returns:
            Generated text string
        """
        if not self._llm:
            raise RuntimeError(self._load_error or "Model not loaded")

        response = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or []
        )

        return response["choices"][0]["text"]

    def generate_stream(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7,
                        stop: Optional[List[str]] = None) -> Generator[str, None, None]:
        """
        Generate text with streaming output.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: List of stop sequences

        Yields:
            Generated text chunks
        """
        if not self._llm:
            raise RuntimeError(self._load_error or "Model not loaded")

        for chunk in self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
            stream=True
        ):
            text = chunk["choices"][0]["text"]
            if text:
                yield text

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512,
             temperature: float = 0.7, system_prompt: Optional[str] = None) -> str:
        """
        Chat completion with message history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt to prepend

        Returns:
            Assistant's response text
        """
        if not self._llm:
            raise RuntimeError(self._load_error or "Model not loaded")

        # Build prompt from messages (auto-detects model format)
        prompt = self._format_chat_prompt(messages, system_prompt)
        stop_tokens = self._get_stop_tokens()

        response = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_tokens
        )

        return response["choices"][0]["text"].strip()

    def chat_stream(self, messages: List[Dict[str, str]], max_tokens: int = 512,
                    temperature: float = 0.7, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """
        Chat completion with streaming output.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt to prepend

        Yields:
            Response text chunks
        """
        if not self._llm:
            raise RuntimeError(self._load_error or "Model not loaded")

        prompt = self._format_chat_prompt(messages, system_prompt)
        stop_tokens = self._get_stop_tokens()

        for chunk in self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_tokens,
            stream=True
        ):
            text = chunk["choices"][0]["text"]
            if text:
                yield text

    def _detect_model_format(self) -> str:
        """
        Detect which prompt format to use based on the model name/path.

        Returns:
            Format type: 'llama3', 'llama2', 'mistral', or 'chatml'
        """
        model_path = (self._model_path or "").lower()
        model_name = (self._model_name or "").lower()

        # Check for Llama 3.x models
        if "llama-3" in model_path or "llama3" in model_path or "llama-3" in model_name:
            return "llama3"

        # Check for Mistral models
        if "mistral" in model_path or "mistral" in model_name:
            return "mistral"

        # Check for TinyLlama (uses ChatML format)
        if "tinyllama" in model_path or "tinyllama" in model_name:
            return "chatml"

        # Default to Llama 2 format for older/unknown models
        return "llama2"

    def _get_stop_tokens(self) -> List[str]:
        """Get appropriate stop tokens for the detected model format."""
        fmt = self._detect_model_format()

        if fmt == "llama3":
            return ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]
        elif fmt == "mistral":
            return ["</s>", "[INST]"]
        elif fmt == "chatml":
            return ["</s>", "<|im_end|>"]
        else:  # llama2
            return ["</s>", "[INST]", "[/INST]"]

    def _format_chat_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """
        Format messages into a prompt string.
        Auto-detects model type and uses appropriate format.

        Args:
            messages: List of message dicts
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt string
        """
        fmt = self._detect_model_format()
        print(f"[LlamaRuntime] Using prompt format: {fmt}")

        if fmt == "llama3":
            return self._format_llama3(messages, system_prompt)
        elif fmt == "mistral":
            return self._format_mistral(messages, system_prompt)
        elif fmt == "chatml":
            return self._format_chatml(messages, system_prompt)
        else:
            return self._format_llama2(messages, system_prompt)

    def _format_llama3(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Format for Llama 3.x models"""
        # Note: llama-cpp-python automatically adds <|begin_of_text|>, so we don't add it here
        # Adding it manually causes "duplicate leading token" warning and reduced quality
        prompt_parts = []

        if system_prompt:
            prompt_parts.append("<|start_header_id|>system<|end_header_id|>\n\n")
            prompt_parts.append(system_prompt)
            prompt_parts.append("<|eot_id|>")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system" and not system_prompt:
                prompt_parts.append("<|start_header_id|>system<|end_header_id|>\n\n")
                prompt_parts.append(content)
                prompt_parts.append("<|eot_id|>")
            elif role == "user":
                prompt_parts.append("<|start_header_id|>user<|end_header_id|>\n\n")
                prompt_parts.append(content)
                prompt_parts.append("<|eot_id|>")
            elif role == "assistant":
                prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
                prompt_parts.append(content)
                prompt_parts.append("<|eot_id|>")

        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(prompt_parts)

    def _format_mistral(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Format for Mistral models"""
        prompt_parts = ["<s>"]

        # Mistral puts system in the first [INST] block
        first_user_content = ""
        if system_prompt:
            first_user_content = system_prompt + "\n\n"

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                first_user_content = content + "\n\n" + first_user_content
            elif role == "user":
                if i == 0 or (i == 1 and messages[0].get("role") == "system"):
                    prompt_parts.append(f"[INST] {first_user_content}{content} [/INST]")
                    first_user_content = ""
                else:
                    prompt_parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                prompt_parts.append(f" {content}</s>")

        return "".join(prompt_parts)

    def _format_chatml(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Format for ChatML models (TinyLlama, etc.)"""
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system" and not system_prompt:
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")

        prompt_parts.append("<|im_start|>assistant\n")
        return "".join(prompt_parts)

    def _format_llama2(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Format for Llama 2 models"""
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system" and not system_prompt:
                prompt_parts.insert(0, f"<<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "user":
                prompt_parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                prompt_parts.append(f" {content} </s>")

        return "".join(prompt_parts)

    def unload(self):
        """Unload the model to free memory"""
        if self._llm:
            del self._llm
            self._llm = None
            print("[LlamaRuntime] Model unloaded")


# Singleton instance
_runtime_instance = None


def get_llama_runtime(model_path: Optional[str] = None) -> LlamaRuntime:
    """Get or create the singleton runtime instance"""
    global _runtime_instance
    if _runtime_instance is None or model_path:
        _runtime_instance = LlamaRuntime(model_path)
    return _runtime_instance


def reload_llama_runtime() -> LlamaRuntime:
    """Force reload of the runtime"""
    global _runtime_instance
    if _runtime_instance:
        _runtime_instance.unload()
    _runtime_instance = None
    return get_llama_runtime()


# Quick test
if __name__ == "__main__":
    runtime = get_llama_runtime()
    print("Status:", runtime.get_status())

    if runtime.is_available():
        response = runtime.chat([
            {"role": "user", "content": "What is the capital of France?"}
        ])
        print("Response:", response)
