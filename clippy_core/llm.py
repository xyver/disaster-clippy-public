"""
LLM abstraction for clippy_core.

Supports multiple LLM providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Local (Ollama, llama.cpp) - via fallback

This module provides both sync and async interfaces.
"""

from typing import List, Dict, Any, Optional, AsyncGenerator, Generator
import os

from .config import ClippyConfig
from .schemas import ChatMessage, ResponseMethod


class LLMService:
    """
    Unified LLM service for response generation.

    Handles provider selection, streaming, and fallback logic.
    """

    def __init__(self, config: Optional[ClippyConfig] = None):
        """
        Initialize LLM service.

        Args:
            config: ClippyConfig instance. If None, uses environment variables.
        """
        self.config = config or ClippyConfig.from_env()
        self._client = None
        self._async_client = None

    def _get_client(self):
        """Get or create sync LLM client."""
        if self._client is not None:
            return self._client

        provider = self.config.llm_provider.lower()

        if provider == "anthropic":
            api_key = self.config.get_anthropic_api_key()
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")

            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key)

        else:  # Default to OpenAI
            api_key = self.config.get_openai_api_key()
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")

            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)

        return self._client

    async def _get_async_client(self):
        """Get or create async LLM client."""
        if self._async_client is not None:
            return self._async_client

        provider = self.config.llm_provider.lower()

        if provider == "anthropic":
            api_key = self.config.get_anthropic_api_key()
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")

            from anthropic import AsyncAnthropic
            self._async_client = AsyncAnthropic(api_key=api_key)

        else:  # Default to OpenAI
            api_key = self.config.get_openai_api_key()
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")

            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(api_key=api_key)

        return self._async_client

    def generate(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response synchronously.

        Args:
            messages: Conversation history as ChatMessage objects
            system_prompt: Optional system prompt override

        Returns:
            Generated response text
        """
        client = self._get_client()
        provider = self.config.llm_provider.lower()

        # Build messages list
        formatted_messages = []

        if provider == "anthropic":
            # Anthropic uses system as separate parameter
            for msg in messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            response = client.messages.create(
                model=self.config.llm_model,
                max_tokens=self.config.llm_max_tokens,
                system=system_prompt or self._get_default_system_prompt(),
                messages=formatted_messages,
            )
            return response.content[0].text

        else:  # OpenAI
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            else:
                formatted_messages.append({
                    "role": "system",
                    "content": self._get_default_system_prompt()
                })

            for msg in messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            response = client.chat.completions.create(
                model=self.config.llm_model,
                messages=formatted_messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
            return response.choices[0].message.content

    async def generate_async(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response asynchronously.

        Args:
            messages: Conversation history as ChatMessage objects
            system_prompt: Optional system prompt override

        Returns:
            Generated response text
        """
        client = await self._get_async_client()
        provider = self.config.llm_provider.lower()

        formatted_messages = []

        if provider == "anthropic":
            for msg in messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            response = await client.messages.create(
                model=self.config.llm_model,
                max_tokens=self.config.llm_max_tokens,
                system=system_prompt or self._get_default_system_prompt(),
                messages=formatted_messages,
            )
            return response.content[0].text

        else:  # OpenAI
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            else:
                formatted_messages.append({
                    "role": "system",
                    "content": self._get_default_system_prompt()
                })

            for msg in messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            response = await client.chat.completions.create(
                model=self.config.llm_model,
                messages=formatted_messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
            return response.choices[0].message.content

    def generate_stream(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response synchronously.

        Args:
            messages: Conversation history
            system_prompt: Optional system prompt override

        Yields:
            Response text chunks
        """
        client = self._get_client()
        provider = self.config.llm_provider.lower()

        formatted_messages = []

        if provider == "anthropic":
            for msg in messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            with client.messages.stream(
                model=self.config.llm_model,
                max_tokens=self.config.llm_max_tokens,
                system=system_prompt or self._get_default_system_prompt(),
                messages=formatted_messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text

        else:  # OpenAI
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            else:
                formatted_messages.append({
                    "role": "system",
                    "content": self._get_default_system_prompt()
                })

            for msg in messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            stream = client.chat.completions.create(
                model=self.config.llm_model,
                messages=formatted_messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    async def generate_stream_async(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response asynchronously.

        Args:
            messages: Conversation history
            system_prompt: Optional system prompt override

        Yields:
            Response text chunks
        """
        client = await self._get_async_client()
        provider = self.config.llm_provider.lower()

        formatted_messages = []

        if provider == "anthropic":
            for msg in messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            async with client.messages.stream(
                model=self.config.llm_model,
                max_tokens=self.config.llm_max_tokens,
                system=system_prompt or self._get_default_system_prompt(),
                messages=formatted_messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        else:  # OpenAI
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            else:
                formatted_messages.append({
                    "role": "system",
                    "content": self._get_default_system_prompt()
                })

            for msg in messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            stream = await client.chat.completions.create(
                model=self.config.llm_model,
                messages=formatted_messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt."""
        return """You are a helpful assistant with access to a knowledge base.

When answering questions:
- Use information from the provided search results
- Cite sources when relevant
- Be concise but thorough
- If the search results don't contain relevant information, say so
- Offer to help refine the search if needed

Focus on providing accurate, actionable information."""
