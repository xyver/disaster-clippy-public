"""
ChatService - the main entry point: search the store, build cited context,
generate an answer.

    from clippy_core import ChatService, ClippyConfig
    from clippy_core.vectordb import SQLiteHybridStore

    config = ClippyConfig(index_path="rules.sqlite")
    chat = ChatService(SQLiteHybridStore(config.index_path), config=config)
    resp = chat.chat_sync("How many jumps are allowed in the short program?",
                          filters={"edition": "2025-26"})
    print(resp.text)
    for r in resp.search_results:
        print(r.citation(), r.url)

Customisation points (no need to edit this file):
- system prompt: ``system_prompt=`` argument, or ``config.prompt_path``
- passage layout: pass ``formatter=`` (subclass of ContextFormatter)
- LLM: pass ``llm_service=`` (subclass of LLMService)
- store: anything with a compatible ``search()``
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from .config import ClippyConfig
from .context import ContextFormatter, extractive_answer
from .llm import LLMService
from .schemas import (ChatMessage, ChatResponse, ResponseMethod, SearchMethod, SearchResponse,
                      SearchResult, SourceInfo)

DEFAULT_PROMPT_FILE = Path(__file__).parent / "prompts" / "default.md"


def load_prompt(path: Optional[str | Path]) -> str:
    return Path(path or DEFAULT_PROMPT_FILE).read_text(encoding="utf-8").strip()


class ChatService:
    def __init__(self, vector_store=None, config: Optional[ClippyConfig] = None,
                 llm_service: Optional[LLMService] = None,
                 formatter: Optional[ContextFormatter] = None,
                 system_prompt: Optional[str] = None):
        self.config = config or ClippyConfig.from_env()
        self.vector_store = vector_store
        self.llm = llm_service or LLMService(self.config)
        self.formatter = formatter or ContextFormatter(self.config.max_context_chars,
                                                       self.config.per_result_chars)
        self.system_prompt = system_prompt or load_prompt(self.config.prompt_path)

    # ------------------------------------------------------------------ search

    async def search(self, query: str, sources: Optional[List[str]] = None, limit: Optional[int] = None,
                     filters: Optional[Dict[str, Any]] = None, mode: Optional[str] = None) -> SearchResponse:
        mode = mode or self.config.search_mode
        limit = limit or self.config.n_results
        if self.vector_store is None:
            return SearchResponse([], query, SearchMethod.KEYWORD, error="No vector store configured")

        kwargs: Dict[str, Any] = {"n_results": limit, "sources": sources}
        params = inspect.signature(self.vector_store.search).parameters
        if "filters" in params:
            kwargs["filters"] = filters
        elif filters:
            return SearchResponse([], query, SearchMethod.SEMANTIC,
                                  error=f"{type(self.vector_store).__name__} does not support filters")
        if "mode" in params:
            kwargs["mode"] = mode

        try:
            if inspect.iscoroutinefunction(self.vector_store.search):
                results = await self.vector_store.search(query, **kwargs)
            else:
                results = await asyncio.to_thread(self.vector_store.search, query, **kwargs)
            results = [SearchResult.from_chromadb(r) if isinstance(r, dict) else r for r in results]
            method = {"keyword": SearchMethod.KEYWORD, "semantic": SearchMethod.SEMANTIC}.get(
                mode, SearchMethod.HYBRID)
            return SearchResponse(results, query, method)
        except Exception as e:  # surface store errors instead of crashing the caller
            return SearchResponse([], query, SearchMethod.HYBRID, error=str(e))

    # ------------------------------------------------------------------ chat

    async def chat(self, message: str, sources: Optional[List[str]] = None,
                   filters: Optional[Dict[str, Any]] = None, system_prompt: Optional[str] = None,
                   conversation_history: Optional[List[ChatMessage]] = None,
                   stream: bool = False, host_context: str = "") -> Union[ChatResponse, AsyncGenerator[str, None]]:
        found = await self.search(message, sources=sources, filters=filters)
        if found.error:
            text = f"Search failed: {found.error}"
            if stream:
                async def one_shot():
                    yield text
                return one_shot()
            return ChatResponse(text=text, method=ResponseMethod.SIMPLE,
                                error=found.error)

        return await self.answer(message, found.results, system_prompt=system_prompt,
                                 conversation_history=conversation_history, stream=stream,
                                 host_context=host_context)

    async def answer(self, message: str, results: List[SearchResult],
                     system_prompt: Optional[str] = None,
                     conversation_history: Optional[List[ChatMessage]] = None,
                     stream: bool = False, host_context: str = "",
                     max_evidence: Optional[int] = None) -> Union[ChatResponse, AsyncGenerator[str, None]]:
        """Answer over prepared passages supplied by a host app, without searching.

        The returned references are exactly the passages included in the model
        context. The host owns source authorization before passing ``results``.
        """
        evidence = results[:max_evidence] if max_evidence is not None else results
        evidence = self.formatter.used(evidence)

        if self.llm.provider == "none":
            evidence = evidence[:5]
            text = extractive_answer(evidence)
            if stream:
                async def one_shot():
                    yield text
                return one_shot()
            return ChatResponse(text=text, method=ResponseMethod.SIMPLE,
                                sources_used=sorted({r.source_id for r in evidence}),
                                search_results=evidence)

        messages = self._build_messages(message, self.formatter.format(evidence),
                                        conversation_history or [], host_context)
        prompt = system_prompt or self.system_prompt

        if stream:
            return self.llm.generate_stream_async(messages, prompt)
        try:
            text = await self.llm.generate_async(messages, prompt)
            return ChatResponse(text=text, method=ResponseMethod.CLOUD_LLM,
                                sources_used=sorted({r.source_id for r in evidence}),
                                search_results=evidence)
        except Exception as e:
            return ChatResponse(text=extractive_answer(evidence), method=ResponseMethod.SIMPLE,
                                search_results=evidence, error=f"LLM failed, showing passages: {e}")

    @staticmethod
    def _build_messages(message: str, context: str, history: List[ChatMessage],
                        host_context: str = "") -> List[ChatMessage]:
        application_context = (f"Application context (for personalization, not source evidence):\n"
                               f"{host_context}\n\n") if host_context else ""
        user = (f"{application_context}Question: {message}\n\n"
                f"Passages from the knowledge base:\n\n{context}\n\n"
                "Answer the question using only these passages, citing them by number.")
        return list(history[-10:]) + [ChatMessage.user(user)]

    # ------------------------------------------------------------------ misc

    async def get_sources(self) -> List[SourceInfo]:
        store = self.vector_store
        if store is None or not hasattr(store, "get_sources"):
            return []
        if inspect.iscoroutinefunction(store.get_sources):
            return await store.get_sources()
        return await asyncio.to_thread(store.get_sources)

    # Synchronous convenience wrappers (for scripts and the CLI)

    def chat_sync(self, message: str, **kwargs) -> ChatResponse:
        kwargs.pop("stream", None)
        return asyncio.run(self.chat(message, stream=False, **kwargs))

    def answer_sync(self, message: str, results: List[SearchResult], **kwargs) -> ChatResponse:
        kwargs.pop("stream", None)
        return asyncio.run(self.answer(message, results, stream=False, **kwargs))

    def search_sync(self, query: str, **kwargs) -> SearchResponse:
        return asyncio.run(self.search(query, **kwargs))
