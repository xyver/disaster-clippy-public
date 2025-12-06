"""
Unified AI Service - Search and Response Generation

Provides a unified interface for:
- Semantic search (online) and keyword search (offline)
- LLM response generation (cloud or local Ollama)
- Automatic fallback handling based on connection mode

This module unifies the search and response pipelines so both
online and offline modes use the same interface.
"""

import os
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from enum import Enum


class SearchMethod(Enum):
    """Search method used"""
    SEMANTIC = "semantic"  # Embedding-based vector search
    KEYWORD = "keyword"    # Simple keyword matching
    LOCAL_SEMANTIC = "local_semantic"  # Future: local embeddings


class ResponseMethod(Enum):
    """Response generation method used"""
    CLOUD_LLM = "cloud_llm"    # OpenAI/Anthropic
    LOCAL_LLM = "local_llm"    # Ollama
    SIMPLE = "simple"          # No LLM, formatted response


@dataclass
class SearchResult:
    """Unified search result"""
    articles: List[Dict[str, Any]]
    method: SearchMethod
    query: str
    error: Optional[str] = None


@dataclass
class ResponseResult:
    """Unified response result"""
    text: str
    method: ResponseMethod
    error: Optional[str] = None


class AIService:
    """
    Unified AI service for search and response generation.

    Provides consistent interface regardless of online/offline mode.
    Handles fallback logic internally.
    """

    def __init__(self):
        self._vector_store = None
        self._llm = None
        self._connection_manager = None

    def _get_vector_store(self):
        """Lazy load vector store"""
        if self._vector_store is None:
            from offline_tools.vectordb import get_vector_store as create_vector_store
            mode = os.getenv("VECTOR_DB_MODE", "local")
            self._vector_store = create_vector_store(mode=mode)
        return self._vector_store

    def _get_connection_manager(self):
        """Lazy load connection manager"""
        if self._connection_manager is None:
            from admin.connection_manager import get_connection_manager
            self._connection_manager = get_connection_manager()
        return self._connection_manager

    def _get_llm(self):
        """Lazy load LLM"""
        if self._llm is None:
            provider = os.getenv("LLM_PROVIDER", "openai").lower()

            try:
                if provider == "anthropic":
                    from langchain_anthropic import ChatAnthropic
                    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
                    self._llm = ChatAnthropic(
                        model=model,
                        temperature=0.7,
                        max_tokens=1024
                    )
                else:
                    from langchain_openai import ChatOpenAI
                    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                    self._llm = ChatOpenAI(
                        model=model,
                        temperature=0.7,
                        max_tokens=1024
                    )
            except Exception as e:
                print(f"Failed to initialize LLM: {e}")
                self._llm = None

        return self._llm

    def search(self, query: str, n_results: int = 10,
               source_filter: Optional[Dict[str, Any]] = None,
               force_method: Optional[SearchMethod] = None) -> SearchResult:
        """
        Unified search function.

        Automatically selects the appropriate search method based on:
        1. Connection mode (online_only, hybrid, offline_only)
        2. Railway proxy availability (for local admins without Pinecone)
        3. Forced method override (for testing)
        4. Fallback on failure

        Args:
            query: Search query
            n_results: Number of results to return
            source_filter: Optional filter dict (ChromaDB format)
            force_method: Override automatic method selection

        Returns:
            SearchResult with articles and metadata
        """
        conn = self._get_connection_manager()

        # Check if we should use Railway proxy for search
        from admin.local_config import get_local_config
        config = get_local_config()

        if config.should_use_proxy_for_search() and not force_method:
            # Use Railway proxy for online search
            result = self._search_via_proxy(query, n_results, source_filter)
            if result and not result.error:
                return result
            # Fallback to local on proxy failure if hybrid mode
            if conn.get_mode() == "hybrid":
                print("Proxy search failed, falling back to local search")
            else:
                return result  # Return error in online_only mode

        # Local search path
        store = self._get_vector_store()

        # Determine method
        if force_method:
            method = force_method
        elif not conn.should_try_online():
            method = SearchMethod.KEYWORD
        else:
            method = SearchMethod.SEMANTIC

        # Execute search with fallback
        try:
            if method == SearchMethod.SEMANTIC:
                articles = store.search(query, n_results=n_results, filter=source_filter)
                conn.on_api_success()
                return SearchResult(
                    articles=articles,
                    method=SearchMethod.SEMANTIC,
                    query=query
                )
        except Exception as e:
            print(f"Semantic search failed: {e}")
            conn.on_api_failure(e)

            # Fallback to keyword if in hybrid mode
            if conn.get_mode() == "hybrid":
                method = SearchMethod.KEYWORD
            else:
                return SearchResult(
                    articles=[],
                    method=SearchMethod.SEMANTIC,
                    query=query,
                    error=str(e)
                )

        # Keyword search (offline or fallback)
        try:
            articles = store.search_offline(query, n_results=n_results, filter=source_filter)
            return SearchResult(
                articles=articles,
                method=SearchMethod.KEYWORD,
                query=query
            )
        except Exception as e:
            return SearchResult(
                articles=[],
                method=SearchMethod.KEYWORD,
                query=query,
                error=str(e)
            )

    def _search_via_proxy(self, query: str, n_results: int = 10,
                          source_filter: Optional[Dict[str, Any]] = None) -> SearchResult:
        """
        Search using Railway proxy.
        Used when local admin doesn't have Pinecone but wants online search.

        Args:
            query: Search query
            n_results: Number of results
            source_filter: ChromaDB filter dict (e.g., {"source": {"$in": [...]}})

        Returns:
            SearchResult from proxy
        """
        try:
            from admin.cloud_proxy import get_proxy_client
            proxy = get_proxy_client()

            if not proxy.is_configured():
                return SearchResult(
                    articles=[],
                    method=SearchMethod.SEMANTIC,
                    query=query,
                    error="Proxy not configured"
                )

            # Convert ChromaDB filter to source list for proxy
            sources = None
            if source_filter and "source" in source_filter:
                source_val = source_filter["source"]
                if isinstance(source_val, dict) and "$in" in source_val:
                    sources = source_val["$in"]

            result = proxy.search(query, n_results=n_results, sources=sources)

            if "error" in result:
                return SearchResult(
                    articles=[],
                    method=SearchMethod.SEMANTIC,
                    query=query,
                    error=result["error"]
                )

            # Convert proxy articles to expected format
            articles = []
            for a in result.get("articles", []):
                articles.append({
                    "id": a.get("title", ""),  # Use title as ID
                    "content": a.get("snippet", ""),
                    "score": a.get("score", 0),
                    "metadata": {
                        "title": a.get("title", "Unknown"),
                        "url": a.get("url", ""),
                        "source": a.get("source", "unknown")
                    }
                })

            return SearchResult(
                articles=articles,
                method=SearchMethod.SEMANTIC,
                query=query
            )

        except Exception as e:
            print(f"Proxy search error: {e}")
            return SearchResult(
                articles=[],
                method=SearchMethod.SEMANTIC,
                query=query,
                error=str(e)
            )

    def generate_response(self, query: str, context: str, history: list,
                          force_method: Optional[ResponseMethod] = None) -> ResponseResult:
        """
        Unified response generation.

        Automatically selects the appropriate method based on:
        1. Connection mode
        2. Ollama availability
        3. Fallback on failure

        Args:
            query: User's question
            context: Formatted article context
            history: Conversation history
            force_method: Override automatic method selection

        Returns:
            ResponseResult with text and metadata
        """
        conn = self._get_connection_manager()

        # Determine method
        if force_method:
            method = force_method
        elif not conn.should_try_online():
            method = ResponseMethod.LOCAL_LLM
        else:
            method = ResponseMethod.CLOUD_LLM

        # Execute with fallback chain: Cloud LLM -> Local LLM -> Simple
        if method == ResponseMethod.CLOUD_LLM:
            result = self._try_cloud_llm(query, context, history)
            if result:
                conn.on_api_success()
                return result

            # Fallback
            conn.on_api_failure()
            if conn.get_mode() == "hybrid":
                method = ResponseMethod.LOCAL_LLM
            else:
                return ResponseResult(
                    text="Unable to connect to cloud AI service. Please check your connection.",
                    method=ResponseMethod.CLOUD_LLM,
                    error="Cloud LLM unavailable"
                )

        if method == ResponseMethod.LOCAL_LLM:
            result = self._try_local_llm(query, context, history)
            if result:
                return result
            # Fallback to simple
            method = ResponseMethod.SIMPLE

        # Simple response (no LLM)
        return self._generate_simple_response(query, context)

    def _try_cloud_llm(self, query: str, context: str, history: list) -> Optional[ResponseResult]:
        """Try to generate response with cloud LLM"""
        try:
            llm = self._get_llm()
            if not llm:
                return None

            prompt = self._build_prompt("online")
            chain = prompt | llm

            response = chain.invoke({
                "query": query,
                "context": context,
                "history": history
            })

            return ResponseResult(
                text=response.content,
                method=ResponseMethod.CLOUD_LLM
            )
        except Exception as e:
            print(f"Cloud LLM error: {e}")
            return None

    def _try_local_llm(self, query: str, context: str, history: list) -> Optional[ResponseResult]:
        """Try to generate response with local Ollama"""
        try:
            from admin.local_config import get_local_config
            config = get_local_config()

            if not config.is_ollama_enabled():
                return None

            from admin.ollama_manager import get_ollama_manager
            ollama = get_ollama_manager()

            # Get system prompt
            system = self._get_system_prompt("offline")

            # Build messages
            messages = []
            for msg in history[-10:]:
                if hasattr(msg, 'content'):
                    role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                    messages.append({"role": role, "content": msg.content})

            # Add current query
            user_message = f"""User query: {query}

Relevant articles from knowledge base:
{context}

Based on these search results, help the user find what they need."""
            messages.append({"role": "user", "content": user_message})

            # Generate
            response = ollama.chat(messages, system=system)
            if response:
                return ResponseResult(
                    text=response,
                    method=ResponseMethod.LOCAL_LLM
                )

            return None
        except Exception as e:
            print(f"Local LLM error: {e}")
            return None

    def _generate_simple_response(self, query: str, context: str) -> ResponseResult:
        """Generate simple response without LLM"""
        if "No relevant articles found" in context:
            text = "I couldn't find any matching articles in my offline database. Try different search terms or check that you have source packs installed."
        else:
            # Extract article titles
            lines = context.split("\n")
            articles = []
            current_title = None

            for line in lines:
                if line.startswith("Article #"):
                    current_title = line.split(": ", 1)[-1] if ": " in line else line
                elif line.startswith("Content Preview:") and current_title:
                    articles.append(current_title)
                    current_title = None

            if articles:
                article_list = "\n".join(f"- {a}" for a in articles[:5])
                text = f"Based on your search for \"{query}\", I found these relevant articles:\n\n{article_list}\n\nCheck the articles panel for more details and links. (Running in offline mode - install a local LLM for conversational responses)"
            else:
                text = "I found some articles that might help. Check the articles panel on the right for details. (Running in offline mode)"

        return ResponseResult(
            text=text,
            method=ResponseMethod.SIMPLE
        )

    def _get_system_prompt(self, mode: str = "online") -> str:
        """Get system prompt from config"""
        try:
            from admin.local_config import get_local_config
            config = get_local_config()
            prompt = config.get_prompt(mode)
            if prompt:
                return prompt
        except Exception:
            pass

        # Defaults
        if mode == "offline":
            return """You are Disaster Clippy, a helpful assistant for DIY and humanitarian resources.
Your role is to help users find relevant articles and answer questions based on the provided context.
Be concise, practical, and helpful. Focus on actionable information.
Only recommend articles that are in the provided context - do not make up articles."""
        else:
            return """You are Disaster Clippy, a helpful assistant that helps people find DIY guides and humanitarian resources.

Your role is to:
1. Understand what the user needs help with
2. Suggest relevant articles from the knowledge base
3. Help them refine their search through conversation
4. Answer follow-up questions about the articles

When presenting search results:
- Summarize what each article is about in 1-2 sentences
- Explain why it might be relevant to their situation
- Offer to find more similar articles or narrow down the search

Be conversational, helpful, and practical. Focus on actionable solutions.

IMPORTANT: You can ONLY recommend articles that are provided to you in the context. Do not make up or hallucinate articles that don't exist in the search results."""

    def _build_prompt(self, mode: str = "online"):
        """Build chat prompt template"""
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        return ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt(mode)),
            MessagesPlaceholder(variable_name="history"),
            ("human", """User query: {query}

Relevant articles from knowledge base:
{context}

Based on these search results, help the user find what they need. If the results don't seem relevant, acknowledge that and suggest how they might refine their search.""")
        ])

    # =========================================================================
    # Streaming support
    # =========================================================================

    def generate_response_stream(self, query: str, context: str,
                                  history: list) -> Generator[str, None, None]:
        """
        Stream response generation (for SSE endpoints).

        Yields chunks of text as they're generated.
        Automatically selects method based on connection mode:
        - online_only: Cloud LLM only
        - hybrid: Cloud LLM with local fallback
        - offline_only: Local LLM or simple response

        Falls back gracefully if primary method fails.
        """
        conn = self._get_connection_manager()
        mode = conn.get_mode()
        yielded_anything = False

        # Try cloud LLM first if online mode
        if conn.should_try_online():
            try:
                for chunk in self._stream_cloud_llm(query, context, history):
                    yielded_anything = True
                    yield chunk
                # Success - we're done
                if yielded_anything:
                    return
            except Exception as e:
                print(f"Cloud streaming failed: {e}")
                conn.on_api_failure(e)
                # Fall through to local if hybrid mode

        # Try local LLM if offline or hybrid fallback
        if mode in ("offline_only", "hybrid"):
            try:
                for chunk in self._stream_local_llm(query, context, history):
                    yielded_anything = True
                    yield chunk
                if yielded_anything:
                    return
            except Exception as e:
                print(f"Local streaming failed: {e}")

        # Final fallback: simple response (no LLM)
        if not yielded_anything:
            result = self._generate_simple_response(query, context)
            yield result.text

    def _stream_cloud_llm(self, query: str, context: str,
                          history: list) -> Generator[str, None, None]:
        """Stream from cloud LLM"""
        try:
            llm = self._get_llm()
            if not llm:
                yield "Unable to connect to AI service."
                return

            prompt = self._build_prompt("online")
            chain = prompt | llm

            # Use streaming
            for chunk in chain.stream({
                "query": query,
                "context": context,
                "history": history
            }):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content

            self._get_connection_manager().on_api_success()

        except Exception as e:
            print(f"Streaming error: {e}")
            self._get_connection_manager().on_api_failure(e)
            yield f"Error generating response: {str(e)}"

    def _stream_local_llm(self, query: str, context: str,
                          history: list) -> Generator[str, None, None]:
        """
        Stream from local Ollama LLM.

        Falls back to simple response if:
        - Ollama is not enabled in config
        - Ollama is not installed/running
        - Streaming fails for any reason
        """
        try:
            from admin.local_config import get_local_config
            config = get_local_config()

            # Check if Ollama is enabled
            if not config.is_ollama_enabled():
                print("Local LLM: Ollama not enabled in config")
                return  # Let caller handle fallback

            from admin.ollama_manager import get_ollama_manager
            ollama = get_ollama_manager()

            # Check if Ollama is available
            if not ollama.is_running() and not ollama.is_installed():
                print("Local LLM: Ollama not installed or running")
                return  # Let caller handle fallback

            system = self._get_system_prompt("offline")

            # Build message history
            messages = []
            for msg in history[-10:]:
                if hasattr(msg, 'content'):
                    role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                    messages.append({"role": role, "content": msg.content})

            # Add current query with context
            user_message = f"""User query: {query}

Relevant articles from knowledge base:
{context}

Based on these search results, help the user find what they need."""
            messages.append({"role": "user", "content": user_message})

            # Stream response
            chunk_count = 0
            for chunk in ollama.chat_stream(messages, system=system):
                chunk_count += 1
                yield chunk

            if chunk_count == 0:
                print("Local LLM: No chunks received from Ollama")
                # Don't yield anything - let caller handle fallback

        except Exception as e:
            print(f"Local LLM streaming error: {e}")
            # Don't yield anything - let caller handle fallback


# Singleton instance
_ai_service = None


def get_ai_service() -> AIService:
    """Get or create the singleton AI service"""
    global _ai_service
    if _ai_service is None:
        _ai_service = AIService()
    return _ai_service
