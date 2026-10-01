"""
LLM abstraction for clippy_core.

Providers: Anthropic, OpenAI. Sync, async and streaming interfaces.
Provider "none" is handled by ChatService (no LLM call; extractive answer).

Adding a provider (e.g. Ollama for offline use): add a branch to
``_client`` / ``_async_client`` and to the four generate methods, or
subclass LLMService and pass it to ChatService(llm_service=...).
"""

from typing import AsyncGenerator, Dict, Generator, List, Optional, Protocol

from .config import ClippyConfig
from .schemas import ChatMessage

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
}

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant with access to a knowledge base.

When answering:
- Use only the information in the provided passages.
- Cite passages by their number, like [1] or [2][3].
- If the passages don't answer the question, say so plainly.
- Be concise but complete."""


class LLMBackend(Protocol):
    """Host-supplied model adapter; cloud and local runtimes use the same seam."""

    provider: str

    async def generate_async(self, messages: List[ChatMessage],
                             system_prompt: Optional[str] = None) -> str: ...

    def generate_stream_async(self, messages: List[ChatMessage],
                              system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]: ...


class LLMService:
    """Unified LLM service for response generation."""

    def __init__(self, config: Optional[ClippyConfig] = None):
        self.config = config or ClippyConfig.from_env()
        self.provider = self.config.resolved_llm_provider()
        self.model = self.config.llm_model or DEFAULT_MODELS.get(self.provider, "")
        self._sync_client = None
        self._async_client_obj = None

    # ---- clients ---------------------------------------------------------

    def _client(self):
        if self._sync_client is None:
            if self.provider == "anthropic":
                from anthropic import Anthropic
                self._sync_client = Anthropic(api_key=self._require_key())
            elif self.provider == "openai":
                from openai import OpenAI
                self._sync_client = OpenAI(api_key=self._require_key())
            else:
                raise ValueError(f"LLM provider {self.provider!r} has no client")
        return self._sync_client

    def _async_client(self):
        if self._async_client_obj is None:
            if self.provider == "anthropic":
                from anthropic import AsyncAnthropic
                self._async_client_obj = AsyncAnthropic(api_key=self._require_key())
            elif self.provider == "openai":
                from openai import AsyncOpenAI
                self._async_client_obj = AsyncOpenAI(api_key=self._require_key())
            else:
                raise ValueError(f"LLM provider {self.provider!r} has no client")
        return self._async_client_obj

    def _require_key(self) -> str:
        key = (self.config.get_anthropic_api_key() if self.provider == "anthropic"
               else self.config.get_openai_api_key())
        if not key:
            raise ValueError(f"No API key set for LLM provider {self.provider!r}")
        return key

    # ---- message formatting ---------------------------------------------

    @staticmethod
    def _as_dicts(messages: List[ChatMessage]) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _openai_messages(self, messages, system_prompt):
        return [{"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}] + self._as_dicts(messages)

    def _anthropic_kwargs(self, messages, system_prompt):
        return dict(
            model=self.model,
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
            system=system_prompt or DEFAULT_SYSTEM_PROMPT,
            messages=self._as_dicts(messages),
        )

    def _openai_kwargs(self, messages, system_prompt, stream=False):
        return dict(
            model=self.model,
            messages=self._openai_messages(messages, system_prompt),
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            stream=stream,
        )

    # ---- generation ------------------------------------------------------

    def generate(self, messages: List[ChatMessage], system_prompt: Optional[str] = None) -> str:
        client = self._client()
        if self.provider == "anthropic":
            resp = client.messages.create(**self._anthropic_kwargs(messages, system_prompt))
            return "".join(b.text for b in resp.content if getattr(b, "type", "text") == "text")
        resp = client.chat.completions.create(**self._openai_kwargs(messages, system_prompt))
        return resp.choices[0].message.content

    async def generate_async(self, messages: List[ChatMessage], system_prompt: Optional[str] = None) -> str:
        client = self._async_client()
        if self.provider == "anthropic":
            resp = await client.messages.create(**self._anthropic_kwargs(messages, system_prompt))
            return "".join(b.text for b in resp.content if getattr(b, "type", "text") == "text")
        resp = await client.chat.completions.create(**self._openai_kwargs(messages, system_prompt))
        return resp.choices[0].message.content

    def generate_stream(self, messages: List[ChatMessage],
                        system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        client = self._client()
        if self.provider == "anthropic":
            with client.messages.stream(**self._anthropic_kwargs(messages, system_prompt)) as stream:
                for text in stream.text_stream:
                    yield text
            return
        for chunk in client.chat.completions.create(**self._openai_kwargs(messages, system_prompt, stream=True)):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_stream_async(self, messages: List[ChatMessage],
                                    system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        client = self._async_client()
        if self.provider == "anthropic":
            async with client.messages.stream(**self._anthropic_kwargs(messages, system_prompt)) as stream:
                async for text in stream.text_stream:
                    yield text
            return
        stream = await client.chat.completions.create(**self._openai_kwargs(messages, system_prompt, stream=True))
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
