"""
ChatService - Main entry point for clippy_core chat functionality.

This is the primary interface for using clippy_core in other projects.
It handles:
- Searching the vector store
- Generating LLM responses
- Source filtering
- Custom prompt injection

Example:
    from clippy_core import ChatService, ClippyConfig
    from clippy_core.vectordb.pgvector import PgVectorStore

    store = PgVectorStore(connection_string="...")
    chat = ChatService(vector_store=store)

    response = await chat.chat(
        message="How do I prepare for a hurricane?",
        sources=["fema-ready"],
        system_prompt="You are a preparedness advisor."
    )
"""

from typing import List, Optional, Union, AsyncGenerator
import asyncio

from .config import ClippyConfig
from .schemas import (
    SearchResult,
    SearchResponse,
    ChatMessage,
    ChatResponse,
    SourceInfo,
    SearchMethod,
    ResponseMethod,
)
from .llm import LLMService


class ChatService:
    """
    Main chat service for clippy_core.

    Provides a unified interface for search and chat, with support for:
    - Custom system prompts (for context injection)
    - Source filtering
    - Streaming responses
    - Conversation history
    """

    def __init__(
        self,
        vector_store=None,
        config: Optional[ClippyConfig] = None,
        llm_service: Optional[LLMService] = None,
    ):
        """
        Initialize ChatService.

        Args:
            vector_store: VectorStore instance (required for search).
                         Can be ChromaDB, Pinecone, pgvector, or any VectorStoreBase.
            config: ClippyConfig instance. If None, uses environment variables.
            llm_service: LLMService instance. If None, creates one from config.
        """
        self.config = config or ClippyConfig.from_env()
        self.vector_store = vector_store
        self.llm = llm_service or LLMService(self.config)

        # Search settings
        self.default_search_limit = 10
        self.max_context_length = 8000  # Max chars for context

    async def chat(
        self,
        message: str,
        sources: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[ChatMessage]] = None,
        stream: bool = False,
    ) -> Union[ChatResponse, AsyncGenerator[str, None]]:
        """
        Main chat entry point.

        Args:
            message: User's question
            sources: List of source IDs to search (None = all sources)
            system_prompt: Custom system prompt for context injection.
                          Use this to add user-specific context (location, preferences, etc.)
            conversation_history: Previous messages for context
            stream: If True, returns async generator of response chunks

        Returns:
            ChatResponse object, or async generator if stream=True
        """
        # Search for relevant documents
        search_response = await self.search(message, sources=sources)

        # Build context from search results
        context = self._build_context(search_response.results)

        # Build messages for LLM
        messages = self._build_messages(
            message=message,
            context=context,
            history=conversation_history or []
        )

        if stream:
            return self._generate_stream(messages, system_prompt, search_response.results)

        # Generate response
        try:
            response_text = await self.llm.generate_async(messages, system_prompt)
            return ChatResponse(
                text=response_text,
                method=ResponseMethod.CLOUD_LLM,
                sources_used=list(set(r.source_id for r in search_response.results)),
                search_results=search_response.results,
            )
        except Exception as e:
            return ChatResponse(
                text=f"Error generating response: {str(e)}",
                method=ResponseMethod.SIMPLE,
                error=str(e),
                search_results=search_response.results,
            )

    async def _generate_stream(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str],
        search_results: List[SearchResult],
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        try:
            async for chunk in self.llm.generate_stream_async(messages, system_prompt):
                yield chunk
        except Exception as e:
            yield f"Error: {str(e)}"

    async def search(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        limit: int = None,
    ) -> SearchResponse:
        """
        Search for relevant documents.

        Args:
            query: Search query
            sources: Optional list of source IDs to filter
            limit: Max results (defaults to self.default_search_limit)

        Returns:
            SearchResponse with results
        """
        if self.vector_store is None:
            return SearchResponse(
                results=[],
                query=query,
                method=SearchMethod.KEYWORD,
                error="No vector store configured"
            )

        limit = limit or self.default_search_limit

        try:
            # Check if vector store has async search
            if hasattr(self.vector_store, "search") and asyncio.iscoroutinefunction(self.vector_store.search):
                results = await self.vector_store.search(query, n_results=limit, sources=sources)
            else:
                # Wrap sync search in executor
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: self._search_sync(query, limit, sources)
                )

            # Ensure results are SearchResult objects
            if results and not isinstance(results[0], SearchResult):
                results = [SearchResult.from_chromadb(r) if isinstance(r, dict) else r for r in results]

            return SearchResponse(
                results=results,
                query=query,
                method=SearchMethod.SEMANTIC,
            )
        except Exception as e:
            return SearchResponse(
                results=[],
                query=query,
                method=SearchMethod.KEYWORD,
                error=str(e),
            )

    def _search_sync(
        self,
        query: str,
        limit: int,
        sources: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Synchronous search wrapper."""
        # Build filter for ChromaDB/existing stores
        filter_dict = None
        if sources:
            filter_dict = {"source": {"$in": sources}}

        # Call the store's search method
        if hasattr(self.vector_store, "search"):
            return self.vector_store.search(query, n_results=limit, filter=filter_dict)
        return []

    async def get_sources(self) -> List[SourceInfo]:
        """Get list of available sources."""
        if self.vector_store is None:
            return []

        try:
            if hasattr(self.vector_store, "get_sources") and asyncio.iscoroutinefunction(self.vector_store.get_sources):
                return await self.vector_store.get_sources()
            elif hasattr(self.vector_store, "get_sources"):
                return self.vector_store.get_sources()
            elif hasattr(self.vector_store, "get_stats"):
                # Fallback for existing stores
                stats = self.vector_store.get_stats()
                sources = stats.get("sources", {})
                return [
                    SourceInfo(id=sid, name=sid, count=count)
                    for sid, count in sources.items()
                ]
        except Exception as e:
            print(f"Error getting sources: {e}")

        return []

    def _build_context(self, results: List[SearchResult]) -> str:
        """Build context string from search results."""
        if not results:
            return "No relevant articles found in the knowledge base."

        context_parts = []
        total_length = 0

        for i, result in enumerate(results, 1):
            # Format each result
            title = result.title or result.metadata.get("title", "Unknown")
            source = result.source_id
            content = result.content[:1500]  # Truncate long content

            part = f"Article #{i}: {title}\nSource: {source}\nContent: {content}\n"

            # Check if adding this would exceed max length
            if total_length + len(part) > self.max_context_length:
                break

            context_parts.append(part)
            total_length += len(part)

        return "\n---\n".join(context_parts)

    def _build_messages(
        self,
        message: str,
        context: str,
        history: List[ChatMessage],
    ) -> List[ChatMessage]:
        """Build messages list for LLM."""
        messages = []

        # Add conversation history (last 10 messages)
        for msg in history[-10:]:
            messages.append(msg)

        # Add current query with context
        user_content = f"""User question: {message}

Relevant information from knowledge base:
{context}

Based on the search results above, please help answer the user's question. If the results don't seem relevant, acknowledge that and suggest how they might refine their search."""

        messages.append(ChatMessage.user(user_content))

        return messages

    # Synchronous convenience methods

    def chat_sync(
        self,
        message: str,
        sources: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> ChatResponse:
        """Synchronous version of chat."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.chat(message, sources, system_prompt, conversation_history, stream=False)
            )
        finally:
            loop.close()

    def search_sync(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        limit: int = None,
    ) -> SearchResponse:
        """Synchronous version of search."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.search(query, sources, limit))
        finally:
            loop.close()
