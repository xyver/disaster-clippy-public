"""
Disaster Clippy - Conversational DIY/Humanitarian Knowledge Search
FastAPI backend with LangChain integration
"""

import os
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Import Anthropic support (optional)
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from offline_tools.vectordb import get_vector_store as create_vector_store, MetadataIndex, DOC_TYPE_GUIDE, DOC_TYPE_ARTICLE, DOC_TYPE_PRODUCT, DOC_TYPE_ACADEMIC
from offline_tools.schemas import get_manifest_file

# Import admin panel
from admin import router as admin_router

# Import source packs API
from admin.routes import sourcepacks_router

# Import ZIM server for offline browsing
from admin.zim_server import router as zim_router

# Import unified AI service and connection manager
from admin.ai_service import get_ai_service, SearchMethod, ResponseMethod
from admin.connection_manager import get_connection_manager, sync_mode_from_config, ConnectionState

# Load environment
load_dotenv()

# Admin API key for protected operations
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")


def verify_admin(x_admin_key: Optional[str] = Header(None)) -> bool:
    """
    Dependency to verify admin API key for protected operations.
    If no ADMIN_API_KEY is configured, allows access (for local dev).
    """
    if not ADMIN_API_KEY:
        # No key configured - allow access (local development mode)
        return True

    if not x_admin_key or x_admin_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing admin API key. Provide X-Admin-Key header."
        )
    return True

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Initialize FastAPI
app = FastAPI(
    title="Disaster Clippy",
    description="Conversational search for DIY and humanitarian resources",
    version="0.1.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Serve admin static files
admin_static = BASE_DIR / "admin" / "static"
if admin_static.exists():
    app.mount("/admin/static", StaticFiles(directory=str(admin_static)), name="admin_static")

# Include admin panel routes
app.include_router(admin_router)

# Include source packs API
app.include_router(sourcepacks_router)

# Include ZIM server for offline browsing
app.include_router(zim_router)


@app.on_event("startup")
async def startup_event():
    """Sync master metadata on app startup"""
    try:
        from offline_tools.packager import sync_master_metadata
        result = sync_master_metadata()
        if result.get("added") or result.get("removed") or result.get("updated"):
            print(f"Master sync: {result['added']} added, {result['removed']} removed, {result['updated']} updated")
        print(f"Sources: {result.get('total', 0)} sources, {result.get('total_documents', 0)} documents")
    except Exception as e:
        print(f"Warning: Could not sync master metadata: {e}")


# Lazy initialization for faster startup
vector_store = None
llm = None

def get_vector_store():
    global vector_store
    if vector_store is None:
        # Uses VECTOR_DB_MODE env var: local, pinecone, or railway
        mode = os.getenv("VECTOR_DB_MODE", "local")
        print(f"Initializing vector store in '{mode}' mode...")
        vector_store = create_vector_store(mode=mode)
        print(f"Vector store ready: {type(vector_store).__name__}")
    return vector_store

def reload_vector_store():
    """Force reload the vector store (called when backup path changes)"""
    global vector_store
    vector_store = None
    return get_vector_store()

def get_llm():
    global llm
    if llm is None:
        provider = os.getenv("LLM_PROVIDER", "openai").lower()

        if provider == "anthropic" and ANTHROPIC_AVAILABLE:
            # Use Claude
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
            llm = ChatAnthropic(
                model=model,
                temperature=0.7,
                max_tokens=1024
            )
        else:
            # Use OpenAI (default)
            if provider == "anthropic" and not ANTHROPIC_AVAILABLE:
                print("Warning: Anthropic requested but not installed. Using OpenAI instead.")
                print("Install with: pip install langchain-anthropic")

            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            llm = ChatOpenAI(
                model=model,
                temperature=0.7,
                max_tokens=1024
            )
    return llm

# Default prompts (fallback if config not available)
DEFAULT_ONLINE_PROMPT = """You are Disaster Clippy, a helpful assistant that helps people find DIY guides and humanitarian resources.

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

DEFAULT_OFFLINE_PROMPT = """You are Disaster Clippy, a helpful assistant for DIY and humanitarian resources.
Your role is to help users find relevant articles and answer questions based on the provided context.
Be concise, practical, and helpful. Focus on actionable information.
Only recommend articles that are in the provided context - do not make up articles."""


def get_system_prompt(mode: str = "online") -> str:
    """Get system prompt from config or use default"""
    try:
        from admin.local_config import get_local_config
        config = get_local_config()
        prompt = config.get_prompt(mode)
        if prompt:
            return prompt
    except Exception as e:
        print(f"Could not load prompt from config: {e}")

    # Fallback to defaults
    return DEFAULT_ONLINE_PROMPT if mode == "online" else DEFAULT_OFFLINE_PROMPT


def get_chat_prompt(mode: str = "online"):
    """Build chat prompt template with current system prompt"""
    return ChatPromptTemplate.from_messages([
        ("system", get_system_prompt(mode)),
        MessagesPlaceholder(variable_name="history"),
        ("human", """User query: {query}

Relevant articles from knowledge base:
{context}

Based on these search results, help the user find what they need. If the results don't seem relevant, acknowledge that and suggest how they might refine their search.""")
    ])

# In-memory session storage (for MVP - use Redis/DB in production)
sessions = {}


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    sources: Optional[List[str]] = None  # List of source IDs to filter by


class ChatResponse(BaseModel):
    response: str
    articles: List[dict]
    session_id: str


class IngestRequest(BaseModel):
    source: str = "appropedia"
    limit: Optional[int] = 100


# Simple API models for external website integration
class SimpleQueryRequest(BaseModel):
    """Simple request for external websites - just send a message"""
    message: str
    session_id: Optional[str] = None
    sources: Optional[List[str]] = None  # List of source IDs to filter by


class SimpleQueryResponse(BaseModel):
    """Simple response for external websites - just the AI response text"""
    response: str
    session_id: str


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the chat interface"""
    template_path = BASE_DIR / "templates" / "index.html"
    return template_path.read_text(encoding='utf-8')


@app.get("/health")
async def health_check():
    """Health check endpoint - lightweight, no DB access"""
    return {
        "status": "healthy",
        "service": "disaster-clippy"
    }


@app.get("/welcome")
async def get_welcome():
    """
    Get a dynamic welcome message based on indexed content.
    Uses _master.json for fast cached stats instead of querying ChromaDB.
    """
    from admin.local_config import get_local_config

    # Try to load from _master.json first (fast, cached)
    local_config = get_local_config()
    backup_folder = local_config.get_backup_folder()

    total_docs = 0
    sources = {}
    topics_set = set()
    last_updated = None

    if backup_folder:
        master_file = Path(backup_folder) / "_master.json"
        if master_file.exists():
            try:
                with open(master_file, 'r', encoding='utf-8') as f:
                    master = json.load(f)
                total_docs = master.get("total_documents", 0)
                sources = master.get("sources", {})
                last_updated = master.get("last_updated")

                # Collect topics from all sources
                for source_id, source_info in sources.items():
                    if isinstance(source_info, dict):
                        source_topics = source_info.get("topics", [])
                        if source_topics:
                            topics_set.update(source_topics)
            except Exception as e:
                print(f"Warning: Could not load _master.json: {e}")

    # Fall back to ChromaDB if _master.json empty or missing
    if total_docs == 0:
        store = get_vector_store()
        stats = store.get_stats()
        total_docs = stats.get("total_documents", 0)
        sources = stats.get("sources", {})

    if total_docs == 0:
        return {
            "message": "Hello! I'm Disaster Clippy. My knowledge base is currently empty - an admin needs to index some content first.",
            "stats": {
                "total_documents": 0,
                "topics": [],
                "sources": []
            }
        }

    # If no topics from _master.json, extract from source manifests
    if not topics_set and backup_folder:
        backup_path = Path(backup_folder)
        for source_id in sources.keys():
            manifest_file = backup_path / source_id / get_manifest_file()
            if manifest_file.exists():
                try:
                    with open(manifest_file, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    tags = manifest.get("tags", [])
                    if tags:
                        topics_set.update(tags)
                except Exception:
                    pass

    # Map tags to display-friendly topic names
    topic_display = {
        "water": "Water", "solar": "Solar Energy", "energy": "Energy",
        "food": "Food", "cooking": "Cooking", "shelter": "Shelter",
        "sanitation": "Sanitation", "medical": "Health", "emergency": "Emergency",
        "agriculture": "Agriculture", "construction": "Construction",
        "compost": "Composting", "cooling": "Cooling", "heating": "Heating",
    }

    topics_list = []
    for tag in sorted(topics_set):
        display = topic_display.get(tag.lower(), tag.replace("-", " ").replace("_", " ").title())
        if display not in topics_list:
            topics_list.append(display)
    topics_list = topics_list[:6]  # Limit to 6

    # Build dynamic welcome message
    if topics_list:
        if len(topics_list) > 1:
            topics_str = ", ".join(topics_list[:-1]) + f", and {topics_list[-1]}"
        else:
            topics_str = topics_list[0]
        message = f"Hello! I'm Disaster Clippy. I have {total_docs} guides available covering topics like {topics_str}.\n\nWhat situation do you need help with?"
    else:
        message = f"Hello! I'm Disaster Clippy. I have {total_docs} DIY guides and resources ready to help you.\n\nWhat situation do you need help with?"

    return {
        "message": message,
        "stats": {
            "total_documents": total_docs,
            "topics": topics_list,
            "sources": list(sources.keys()),
            "last_updated": last_updated
        }
    }


@app.get("/sources")
async def get_sources():
    """
    Get list of available sources with document counts.

    IMPORTANT: Uses ChromaDB as source of truth (what's actually searchable),
    with _manifest.json files for display names only.
    """
    from admin.local_config import get_local_config

    # ChromaDB is the source of truth for searchable sources
    store = get_vector_store()
    stats = store.get_stats()
    sources_counts = stats.get("sources", {})
    total_docs = stats.get("total_documents", 0)

    # Load source names from _manifest.json files (for display only)
    local_config = get_local_config()
    backup_folder = local_config.get_backup_folder()

    source_names = {}
    if backup_folder:
        backup_path = Path(backup_folder)
        if backup_path.exists():
            for source_id in sources_counts.keys():
                source_folder = backup_path / source_id
                manifest_file = source_folder / get_manifest_file()
                if manifest_file.exists():
                    try:
                        with open(manifest_file) as f:
                            source_data = json.load(f)
                            source_names[source_id] = source_data.get("name", source_id)
                    except Exception:
                        pass

    # Build sources dict with names and counts
    sources = {}
    for source_id, count in sources_counts.items():
        sources[source_id] = {
            "name": source_names.get(source_id, source_id.replace("_", " ").replace("-", " ").title()),
            "count": count
        }

    return {
        "sources": sources,
        "total": total_docs
    }


def get_connection_mode() -> str:
    """Get current connection mode from local config"""
    try:
        from admin.local_config import get_local_config
        config = get_local_config()
        return config.get_offline_mode()
    except Exception:
        return "hybrid"  # Default to hybrid


def search_articles(query: str, n_results: int = 10,
                   source_filter: dict = None, mode: str = None) -> list:
    """
    Search for articles using the unified AI service.

    Uses the appropriate method based on connection mode:
    - online_only: Semantic search with embedding API
    - hybrid: Semantic search with keyword fallback
    - offline_only: Keyword search only

    Args:
        query: Search query
        n_results: Number of results
        source_filter: ChromaDB filter dict
        mode: Connection mode override (optional)

    Returns:
        List of matching articles
    """
    # Sync mode if provided
    if mode:
        conn = get_connection_manager()
        conn.set_mode(mode)
    else:
        sync_mode_from_config()

    # Use unified AI service
    ai_service = get_ai_service()
    result = ai_service.search(query, n_results=n_results, source_filter=source_filter)

    return result.articles


def generate_response(query: str, context: str, history: list, mode: str = None) -> str:
    """
    Generate a response using the unified AI service.

    Uses the appropriate method based on connection mode:
    - online_only: Cloud LLM (OpenAI/Anthropic)
    - hybrid: Cloud LLM with local fallback
    - offline_only: Local Ollama or simple response

    Args:
        query: User's question
        context: Formatted article context
        history: Conversation history
        mode: Connection mode override (optional)

    Returns:
        Response text
    """
    # Sync mode if provided
    if mode:
        conn = get_connection_manager()
        conn.set_mode(mode)
    else:
        sync_mode_from_config()

    # Use unified AI service
    ai_service = get_ai_service()
    result = ai_service.generate_response(query, context, history)

    return result.text


# Legacy functions for backward compatibility
# These now delegate to the unified AI service

def generate_offline_response(query: str, context: str, history: list) -> str:
    """
    Generate response using local Ollama if available, otherwise simple format.
    Delegates to unified AI service.
    """
    ai_service = get_ai_service()
    result = ai_service.generate_response(
        query, context, history,
        force_method=ResponseMethod.LOCAL_LLM
    )
    return result.text


def format_offline_response(query: str, context: str) -> str:
    """
    Generate a simple response without LLM for offline mode.
    Delegates to unified AI service.
    """
    ai_service = get_ai_service()
    result = ai_service._generate_simple_response(query, context)
    return result.text


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Handles conversational queries with context from vector store.
    Respects connection mode setting (online_only, hybrid, offline_only).
    """
    session_id = request.session_id or datetime.utcnow().isoformat()
    message = request.message.strip()

    # Get connection mode
    mode = get_connection_mode()

    # Get or create session
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "last_results": []
        }

    session = sessions[session_id]

    # Detect if user has a preference for doc type (articles, products, guides)
    preferred_doc_type = detect_doc_type_preference(message)

    # Check for "more like this" type queries
    if any(phrase in message.lower() for phrase in ["more like", "similar to", "like #", "like number"]):
        # Try to find which article they're referring to
        articles = handle_similarity_query(message, session["last_results"])
    else:
        # Search using appropriate method based on connection mode
        # request.sources: None = all, [] = none, [ids] = specific
        source_filter = None
        if request.sources is not None:
            if len(request.sources) == 0:
                # Empty list = no sources selected, return empty results
                articles = []
            else:
                source_filter = {"source": {"$in": request.sources}}
                articles = search_articles(message, n_results=10,
                                          source_filter=source_filter, mode=mode)
        else:
            articles = search_articles(message, n_results=10,
                                       source_filter=source_filter, mode=mode)

    # Prioritize results by doc_type (guides by default, unless user asked for something else)
    articles = prioritize_results_by_doc_type(articles, preferred_doc_type)

    # Filter by sources if specified (post-filter for similarity queries)
    if request.sources is not None and len(request.sources) > 0:
        articles = [a for a in articles if a.get("metadata", {}).get("source") in request.sources]

    # Return top 5 after re-ranking
    articles = articles[:5]

    # Store results for follow-up queries
    session["last_results"] = articles

    # Format context for LLM
    context = format_articles_for_context(articles)

    # Build conversation history (last 10 exchanges)
    history = session["history"][-20:]  # 20 messages = 10 exchanges

    # Generate response using appropriate method based on connection mode
    response_text = generate_response(message, context, history, mode)

    # Update history
    session["history"].append(HumanMessage(content=message))
    session["history"].append(AIMessage(content=response_text))

    # Format articles for response (convert ZIM URLs to local browsable URLs)
    formatted_articles = []
    for a in articles:
        url = a["metadata"].get("url", "")
        # Convert zim:// URLs to local /zim/ URLs for offline browsing
        if url.startswith("zim://"):
            url = _convert_zim_url(url, a["metadata"].get("source", "")) or url
        formatted_articles.append({
            "title": a["metadata"].get("title", "Unknown"),
            "url": url,
            "source": a["metadata"].get("source", "unknown"),
            "doc_type": a.get("doc_type", "article"),
            "score": round(a.get("original_score", a["score"]), 3),
            "snippet": a["content"][:300] + "..." if len(a["content"]) > 300 else a["content"]
        })

    return ChatResponse(
        response=response_text,
        articles=formatted_articles,
        session_id=session_id
    )


# =============================================================================
# EXTERNAL API - Simple endpoint for other websites to embed chat
# =============================================================================

@app.post("/api/v1/chat", response_model=SimpleQueryResponse)
async def simple_chat(request: SimpleQueryRequest):
    """
    Simple chat API for external websites.

    Just send a message, get a response. No articles sidebar, no complex data.
    Perfect for embedding a simple chat widget on other sites.
    Uses unified AI service for consistent behavior.

    Example:
        POST /api/v1/chat
        {"message": "How do I purify water?"}

        Response:
        {"response": "Here are some methods...", "session_id": "abc123"}
    """
    session_id = request.session_id or datetime.utcnow().isoformat()
    message = request.message.strip()

    if not message:
        return SimpleQueryResponse(
            response="Please enter a question about DIY, survival, or humanitarian topics.",
            session_id=session_id
        )

    # Sync connection mode from config
    sync_mode_from_config()

    # Get or create session
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "last_results": []
        }

    session = sessions[session_id]

    # Detect doc type preference and search using unified service
    preferred_doc_type = detect_doc_type_preference(message)

    # Build source filter
    source_filter = None
    if request.sources is not None:
        if len(request.sources) == 0:
            articles = []
        else:
            source_filter = {"source": {"$in": request.sources}}

    if any(phrase in message.lower() for phrase in ["more like", "similar to", "like #", "like number"]):
        articles = handle_similarity_query(message, session["last_results"])
        # Post-filter for similarity queries
        if request.sources is not None and len(request.sources) > 0:
            articles = [a for a in articles if a.get("metadata", {}).get("source") in request.sources]
    elif request.sources is not None and len(request.sources) == 0:
        articles = []  # No sources selected
    else:
        articles = search_articles(message, n_results=10, source_filter=source_filter)

    # Prioritize and limit results
    articles = prioritize_results_by_doc_type(articles, preferred_doc_type)
    articles = articles[:5]
    session["last_results"] = articles

    # Format context for LLM
    context = format_articles_for_context(articles)

    # Build conversation history
    history = session["history"][-20:]

    # Generate response using unified service
    response_text = generate_response(message, context, history)

    # Update history
    session["history"].append(HumanMessage(content=message))
    session["history"].append(AIMessage(content=response_text))

    return SimpleQueryResponse(
        response=response_text,
        session_id=session_id
    )


@app.post("/api/v1/chat/stream")
async def stream_chat(request: SimpleQueryRequest):
    """
    Streaming chat API - returns Server-Sent Events (SSE) for real-time response.

    Use this endpoint for a "typing" effect where tokens appear as they're generated.
    Connect using EventSource in JavaScript.

    Example JavaScript:
        const eventSource = new EventSource('/api/v1/chat/stream?message=...');
        eventSource.onmessage = (e) => {
            if (e.data === '[DONE]') {
                eventSource.close();
            } else {
                appendToChat(e.data);
            }
        };
    """
    session_id = request.session_id or datetime.utcnow().isoformat()
    message = request.message.strip()

    if not message:
        async def empty_response():
            yield "data: Please enter a question.\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty_response(), media_type="text/event-stream")

    # Sync connection mode from config
    sync_mode_from_config()

    # Get or create session
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "last_results": []
        }

    session = sessions[session_id]

    # Search for articles with source filter
    preferred_doc_type = detect_doc_type_preference(message)

    # Build source filter
    source_filter = None
    if request.sources is not None:
        if len(request.sources) == 0:
            # Empty list = no sources selected
            articles = []
        else:
            source_filter = {"source": {"$in": request.sources}}

    if any(phrase in message.lower() for phrase in ["more like", "similar to", "like #", "like number"]):
        articles = handle_similarity_query(message, session["last_results"])
        # Post-filter for similarity queries
        if request.sources is not None and len(request.sources) > 0:
            articles = [a for a in articles if a.get("metadata", {}).get("source") in request.sources]
    elif request.sources is not None and len(request.sources) == 0:
        articles = []  # No sources selected
    else:
        articles = search_articles(message, n_results=10, source_filter=source_filter)

    articles = prioritize_results_by_doc_type(articles, preferred_doc_type)
    articles = articles[:5]
    session["last_results"] = articles

    context = format_articles_for_context(articles)
    history = session["history"][-20:]

    # Get AI service for streaming
    ai_service = get_ai_service()

    async def generate():
        # Send articles first as JSON (prefixed with [ARTICLES])
        # Convert ZIM URLs to local browsable URLs
        def get_browsable_url(a):
            url = a.get("metadata", {}).get("url", "")
            if url.startswith("zim://"):
                return _convert_zim_url(url, a.get("metadata", {}).get("source", "")) or url
            return url

        articles_json = json.dumps([{
            "title": a.get("metadata", {}).get("title", "Unknown"),
            "url": get_browsable_url(a),
            "source": a.get("metadata", {}).get("source", "unknown"),
            "snippet": a.get("content", "")[:200] + "..." if a.get("content", "") else "",
            "score": a.get("score", 0)
        } for a in articles])
        yield f"data: [ARTICLES]{articles_json}\n\n"

        full_response = []
        try:
            for chunk in ai_service.generate_response_stream(message, context, history):
                full_response.append(chunk)
                # Escape any newlines in the chunk for SSE format
                safe_chunk = chunk.replace("\n", "\\n")
                yield f"data: {safe_chunk}\n\n"

            # Signal completion
            yield "data: [DONE]\n\n"

            # Update history with full response
            response_text = "".join(full_response)
            session["history"].append(HumanMessage(content=message))
            session["history"].append(AIMessage(content=response_text))

        except Exception as e:
            yield f"data: [ERROR]{str(e)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/v1/connection-status")
async def get_connection_status():
    """
    Get current connection status for frontend display.

    Returns:
        - mode: User's selected mode (online_only, hybrid, offline_only)
        - is_online: Whether we're currently connected
        - effective_mode: What mode is actually being used
    """
    conn = get_connection_manager()
    sync_mode_from_config()
    return conn.get_status()


@app.post("/api/v1/ping")
async def ping_connectivity():
    """
    Trigger a connectivity check and return status.
    Called by frontend to verify connection.
    """
    conn = get_connection_manager()
    sync_mode_from_config()
    conn.perform_scheduled_ping()
    return conn.get_status()


@app.get("/api/v1/embed")
async def get_embed_code():
    """
    Returns example HTML/JavaScript code for embedding the chat widget
    on external websites.
    """
    # Get the base URL from environment or use default
    base_url = os.getenv("BASE_URL", "https://your-railway-url.railway.app")

    embed_html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Disaster Clippy Chat Widget</title>
    <style>
        #clippy-chat {{
            max-width: 500px;
            margin: 20px auto;
            font-family: Arial, sans-serif;
        }}
        #clippy-messages {{
            border: 1px solid #ccc;
            height: 300px;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 10px;
            background: #f9f9f9;
        }}
        .clippy-msg {{
            margin: 8px 0;
            padding: 8px 12px;
            border-radius: 8px;
        }}
        .clippy-user {{
            background: #007bff;
            color: white;
            text-align: right;
        }}
        .clippy-bot {{
            background: #e9ecef;
            color: #333;
        }}
        #clippy-input {{
            width: calc(100% - 70px);
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }}
        #clippy-send {{
            width: 60px;
            padding: 10px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        #clippy-send:hover {{
            background: #218838;
        }}
    </style>
</head>
<body>

<div id="clippy-chat">
    <h3>Ask Disaster Clippy</h3>
    <div id="clippy-messages"></div>
    <input type="text" id="clippy-input" placeholder="Ask about DIY, survival, emergency prep...">
    <button id="clippy-send">Send</button>
</div>

<script>
const CLIPPY_API = "{base_url}/api/v1/chat";
let clippySessionId = null;

async function sendToClippy() {{
    const input = document.getElementById("clippy-input");
    const messages = document.getElementById("clippy-messages");
    const message = input.value.trim();

    if (!message) return;

    // Show user message
    messages.innerHTML += '<div class="clippy-msg clippy-user">' + message + '</div>';
    input.value = "";
    messages.scrollTop = messages.scrollHeight;

    try {{
        const response = await fetch(CLIPPY_API, {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify({{
                message: message,
                session_id: clippySessionId
            }})
        }});

        const data = await response.json();
        clippySessionId = data.session_id;

        // Show bot response
        messages.innerHTML += '<div class="clippy-msg clippy-bot">' + data.response + '</div>';
        messages.scrollTop = messages.scrollHeight;
    }} catch (error) {{
        messages.innerHTML += '<div class="clippy-msg clippy-bot">Sorry, I could not connect. Please try again.</div>';
    }}
}}

document.getElementById("clippy-send").addEventListener("click", sendToClippy);
document.getElementById("clippy-input").addEventListener("keypress", function(e) {{
    if (e.key === "Enter") sendToClippy();
}});
</script>

</body>
</html>'''

    return {
        "description": "Example HTML code to embed Disaster Clippy chat on your website",
        "api_endpoint": f"{base_url}/api/v1/chat",
        "method": "POST",
        "request_format": {
            "message": "string (required) - The user's question",
            "session_id": "string (optional) - Session ID for conversation continuity"
        },
        "response_format": {
            "response": "string - The AI's response",
            "session_id": "string - Session ID to use for follow-up messages"
        },
        "embed_html": embed_html,
        "notes": [
            "CORS is enabled - you can call this API from any domain",
            "Session IDs maintain conversation history for follow-up questions",
            "No API key required for basic usage"
        ]
    }


@app.post("/ingest")
async def ingest_content(request: IngestRequest, _admin: bool = Depends(verify_admin)):
    """
    Trigger content ingestion from a source.
    For MVP, supports Appropedia.
    Requires admin API key.
    """
    from offline_tools.scraper import ApropediaScraper

    if request.source.lower() != "appropedia":
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown source: {request.source}"}
        )

    # Scrape content
    scraper = ApropediaScraper(rate_limit=1.0)
    pages = scraper.scrape_all(limit=request.limit)

    if not pages:
        return {"status": "no_content", "message": "No pages were scraped"}

    # Convert to documents and add to vector store
    documents = [page.to_dict() for page in pages]
    count = get_vector_store().add_documents(documents)

    return {
        "status": "success",
        "documents_added": count,
        "source": request.source
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return get_vector_store().get_stats()


@app.get("/scan")
async def scan_database():
    """
    Scan database and compare with config to show sync status.
    """
    import json
    from offline_tools.scraper import ApropediaScraper

    config_path = BASE_DIR / "ingest_config.json"
    if not config_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "ingest_config.json not found"}
        )

    with open(config_path) as f:
        config = json.load(f)

    store = get_vector_store()
    stats = store.get_stats()

    # Get all documents from DB
    try:
        all_docs = store.collection.get(include=["metadatas"])
        doc_titles = set()
        doc_sources = {}

        for metadata in all_docs.get("metadatas", []):
            title = metadata.get("title", "Unknown")
            source = metadata.get("source", "unknown")
            doc_titles.add(title)
            doc_sources[source] = doc_sources.get(source, 0) + 1
    except:
        doc_titles = set()
        doc_sources = {}

    # Analyze config sources
    sources = config.get("sources", [])
    enabled_sources = [s for s in sources if s.get("enabled", True)]

    scraper = ApropediaScraper(rate_limit=2.0)
    source_status = []

    for source in enabled_sources:
        source_type = source.get("type")
        limit = source.get("limit", 100)

        if source_type == "search":
            query = source.get("query", "")
            titles = scraper.search_pages(query, limit=limit)
            existing = len([t for t in titles if t in doc_titles])
            new = len(titles) - existing

            source_status.append({
                "type": "search",
                "query": query,
                "limit": limit,
                "found": len(titles),
                "in_database": existing,
                "new_to_add": new
            })

        elif source_type == "category":
            name = source.get("name", "")
            source_status.append({
                "type": "category",
                "name": name,
                "limit": limit,
                "status": "scan not implemented for categories"
            })

    return {
        "database": {
            "total_documents": stats["total_documents"],
            "sources": doc_sources
        },
        "config_sources": source_status,
        "recommendation": "Run POST /sync to add new content" if any(s.get("new_to_add", 0) > 0 for s in source_status) else "Database is up to date"
    }


@app.post("/sync")
async def sync_from_config(clear: bool = False, smart: bool = True, _admin: bool = Depends(verify_admin)):
    """
    Sync database from ingest_config.json.
    Use the same config locally and on Railway for consistent data.

    - smart=true (default): Only scrape and embed NEW pages not in DB
    - smart=false: Re-scrape everything (useful if content changed)
    - clear=true: Clear DB first, then sync all
    """
    import json
    from offline_tools.scraper import ApropediaScraper

    config_path = BASE_DIR / "ingest_config.json"
    if not config_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "ingest_config.json not found"}
        )

    with open(config_path) as f:
        config = json.load(f)

    sources = [s for s in config.get("sources", []) if s.get("enabled", True)]
    if not sources:
        return {"status": "no_sources", "message": "No enabled sources in config"}

    store = get_vector_store()

    if clear:
        store.delete_all()
        existing_titles = set()
    elif smart:
        existing_titles = store.get_existing_titles()
    else:
        existing_titles = set()

    total_added = 0
    total_skipped = 0
    results = []

    for source in sources:
        source_type = source.get("type")
        limit = source.get("limit", 100)

        if source_type == "search":
            query = source.get("query")
            if not query:
                continue

            scraper = ApropediaScraper(rate_limit=1.0)
            urls = scraper.search_pages(query, limit=limit)

            # Filter to only new pages (by title extracted from URL)
            if smart and not clear:
                new_urls = []
                for url in urls:
                    title = url.split("/wiki/")[-1].replace("_", " ")
                    if title not in existing_titles:
                        new_urls.append(url)
                skipped = len(urls) - len(new_urls)
                total_skipped += skipped
                urls = new_urls
            else:
                skipped = 0

            pages = []
            for url in urls:
                page = scraper.scrape_page(url)
                if page:
                    pages.append(page)

            if pages:
                documents = [page.to_dict() for page in pages]
                count = store.add_documents(documents)
                total_added += count
                results.append({
                    "type": "search",
                    "query": query,
                    "added": count,
                    "skipped": skipped if smart else 0
                })

        elif source_type == "category":
            name = source.get("name")
            if not name:
                continue

            scraper = ApropediaScraper(rate_limit=1.0)
            scraper.categories = [name]
            urls = scraper.get_page_urls(limit=limit)

            # Filter to only new URLs
            skipped = 0
            if smart and not clear:
                new_urls = []
                for url in urls:
                    title = url.split("/wiki/")[-1].replace("_", " ")
                    if title not in existing_titles:
                        new_urls.append(url)
                skipped = len(urls) - len(new_urls)
                total_skipped += skipped
                urls = new_urls

            pages = []
            for url in urls:
                page = scraper.scrape_page(url)
                if page:
                    pages.append(page)

            if pages:
                documents = [page.to_dict() for page in pages]
                count = store.add_documents(documents)
                total_added += count
                results.append({
                    "type": "category",
                    "name": name,
                    "added": count,
                    "skipped": skipped
                })

    return {
        "status": "success",
        "smart_mode": smart,
        "total_added": total_added,
        "total_skipped": total_skipped,
        "sources_processed": len(results),
        "details": results,
        "total_documents": store.get_stats()["total_documents"]
    }


@app.delete("/clear")
async def clear_database(_admin: bool = Depends(verify_admin)):
    """Clear all indexed content (use with caution!). Requires admin API key."""
    get_vector_store().delete_all()
    return {"status": "cleared"}


@app.get("/export")
async def export_database(_admin: bool = Depends(verify_admin)):
    """
    Export the database as a downloadable zip file.
    Use this to backup or transfer the database.
    Requires admin API key.
    """
    import shutil
    import tempfile
    from fastapi.responses import FileResponse

    db_path = BASE_DIR / "data" / "chroma"
    if not db_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Database not found"}
        )

    # Create a zip file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        zip_path = tmp.name

    shutil.make_archive(zip_path[:-4], 'zip', db_path)

    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename='disaster-clippy-db.zip'
    )


# Helper functions

def detect_doc_type_preference(message: str) -> Optional[str]:
    """
    Detect if user is asking specifically for articles, products, guides, or academic papers.
    Returns the preferred doc_type or None if no preference detected.
    """
    message_lower = message.lower()

    # Academic-related queries (research, papers, studies)
    academic_keywords = [
        "research", "study", "studies", "paper", "papers", "academic",
        "scientific", "journal", "peer-reviewed", "scholarly", "literature",
        "evidence", "data shows", "findings", "experiment", "methodology"
    ]
    if any(kw in message_lower for kw in academic_keywords):
        return DOC_TYPE_ACADEMIC

    # Product-related queries
    product_keywords = [
        "buy", "purchase", "where can i get", "where to buy", "product",
        "manufacturer", "vendor", "commercial", "ready-made", "pre-built",
        "store", "shop", "order", "price", "cost of buying"
    ]
    if any(kw in message_lower for kw in product_keywords):
        return DOC_TYPE_PRODUCT

    # Article-related queries (information, not instructions)
    article_keywords = [
        "what is", "explain", "tell me about", "information about",
        "article about", "learn about", "overview of", "background on",
        "history of", "science behind", "how does it work", "why does"
    ]
    if any(kw in message_lower for kw in article_keywords):
        return DOC_TYPE_ARTICLE

    # Guide-related queries (explicit requests for instructions)
    guide_keywords = [
        "how to", "how do i", "build", "make", "construct", "diy",
        "instructions", "step by step", "tutorial", "guide", "plans"
    ]
    if any(kw in message_lower for kw in guide_keywords):
        return DOC_TYPE_GUIDE

    # No explicit preference - will default to prioritizing guides
    return None


# Cache for source base_urls
_source_base_urls = {}


def _get_source_base_url(source_id: str) -> Optional[str]:
    """Get the base_url for a source from _manifest.json or known mappings"""
    global _source_base_urls

    if source_id in _source_base_urls:
        return _source_base_urls[source_id]

    # Known ZIM sources (Kiwix convention)
    known_urls = {
        "bitcoin": "https://en.bitcoin.it/wiki/",
        "wikipedia": "https://en.wikipedia.org/wiki/",
        "wiktionary": "https://en.wiktionary.org/wiki/",
        "wikihow": "https://www.wikihow.com/",
        "stackexchange": "https://stackexchange.com/",
        "stackoverflow": "https://stackoverflow.com/",
        "gutenberg": "https://www.gutenberg.org/",
    }

    if source_id in known_urls:
        _source_base_urls[source_id] = known_urls[source_id]
        return known_urls[source_id]

    # Try to load from _manifest.json in backup folder
    try:
        local_config = get_local_config()
        backup_folder = local_config.get_backup_folder()

        if backup_folder:
            manifest_file = Path(backup_folder) / source_id / get_manifest_file()
            if manifest_file.exists():
                with open(manifest_file) as f:
                    source_data = json.load(f)
                    base_url = source_data.get("base_url")
                    if base_url:
                        _source_base_urls[source_id] = base_url
                        return base_url
    except Exception:
        pass

    return None


def _convert_zim_url(zim_url: str, source_id: str, prefer_offline: bool = True) -> Optional[str]:
    """
    Convert a zim:// URL to a browsable URL.

    In offline/hybrid mode (default): Returns local ZIM server URL for offline browsing
    In online mode with prefer_offline=False: Returns original web URL if base_url is known

    Args:
        zim_url: URL in format zim://{source_id}/{article_path}
        source_id: The source identifier
        prefer_offline: If True, always return local ZIM server URL (default)

    Returns:
        Local /zim/ URL for offline browsing, or web URL, or None if can't convert
    """
    if not zim_url.startswith('zim://'):
        return zim_url

    # Parse zim:// URL
    # Format: zim://{source_id}/{article_path}
    parts = zim_url[6:].split('/', 1)
    if len(parts) < 2:
        return None

    zim_source = parts[0]
    article_path = parts[1]

    # Always return local ZIM server URL for offline browsing
    # This enables seamless offline experience
    if prefer_offline:
        return f"/zim/{zim_source}/{article_path}"

    # Online fallback: try to get original web URL
    base_url = _get_source_base_url(zim_source) or _get_source_base_url(source_id)
    if base_url:
        if not base_url.endswith('/'):
            base_url += '/'
        article_path = article_path.lstrip('/')
        return base_url + article_path

    # Fall back to local ZIM server
    return f"/zim/{zim_source}/{article_path}"


def prioritize_results_by_doc_type(articles: List[dict],
                                    preferred_type: Optional[str] = None) -> List[dict]:
    """
    Re-rank search results to prioritize guides (unless user asked for something else).

    Args:
        articles: List of search results with metadata
        preferred_type: User's preferred doc_type (None means prioritize guides)

    Returns:
        Re-ranked list of articles
    """
    if not articles:
        return articles

    # Get metadata index for doc_type lookup
    index = MetadataIndex()

    # Determine which type to prioritize
    priority_type = preferred_type if preferred_type else DOC_TYPE_GUIDE

    # Assign priority scores based on doc_type
    scored_articles = []
    for article in articles:
        doc_id = article.get("id", "")
        title = article.get("metadata", {}).get("title", "")

        # Look up doc_type from metadata index
        doc_type = None
        for source in index.list_sources():
            source_data = index._load_source(source)
            if doc_id in source_data["documents"]:
                doc_type = source_data["documents"][doc_id].get("doc_type", DOC_TYPE_ARTICLE)
                break

        # If not found in index, default to article
        if doc_type is None:
            doc_type = DOC_TYPE_ARTICLE

        # Calculate priority boost
        # Priority type gets a significant boost, others get none
        priority_boost = 0.15 if doc_type == priority_type else 0

        # Slight penalty for non-priority types to ensure ordering
        if doc_type != priority_type:
            priority_boost = -0.05

        # Combine with original score
        original_score = article.get("score", 0)
        adjusted_score = original_score + priority_boost

        scored_articles.append({
            **article,
            "original_score": original_score,
            "score": adjusted_score,
            "doc_type": doc_type
        })

    # Sort by adjusted score (descending)
    scored_articles.sort(key=lambda x: x["score"], reverse=True)

    return scored_articles


def format_articles_for_context(articles: List[dict]) -> str:
    """Format search results for LLM context"""
    if not articles:
        return "No relevant articles found in the knowledge base."

    formatted = []
    for i, article in enumerate(articles, 1):
        metadata = article["metadata"]
        content_preview = article["content"][:500]
        doc_type = article.get("doc_type", "article")
        doc_type_label = {
            DOC_TYPE_GUIDE: "Guide (step-by-step instructions)",
            DOC_TYPE_ARTICLE: "Article (general information)",
            DOC_TYPE_PRODUCT: "Product (commercial item)",
            DOC_TYPE_ACADEMIC: "Academic (research paper/study)"
        }.get(doc_type, "Article")

        # Handle ZIM URLs - convert to real URLs if base_url is known
        url = metadata.get('url', 'N/A')
        if url.startswith('zim://'):
            # Try to convert zim:// URL to real URL
            # Format: zim://{source_id}/{article_path}
            real_url = _convert_zim_url(url, metadata.get('source', ''))
            if real_url:
                url_line = f"URL: {real_url}"
            else:
                url_line = "URL: (offline archive)"
        else:
            url_line = f"URL: {url}"

        formatted.append(f"""
Article #{i}: {metadata.get('title', 'Unknown Title')}
Type: {doc_type_label}
Source: {metadata.get('source', 'unknown')}
{url_line}
Categories: {', '.join(metadata.get('categories', [])[:5])}
Relevance Score: {article.get('original_score', article['score']):.2f}

Content Preview:
{content_preview}...
---""")

    return "\n".join(formatted)


def handle_similarity_query(message: str, last_results: List[dict]) -> List[dict]:
    """Handle 'more like this' type queries"""
    import re

    # Try to extract article number
    match = re.search(r'#?(\d+)', message)
    if match and last_results:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(last_results):
            # Find similar to that article
            doc_id = last_results[idx]["id"]
            return get_vector_store().search_similar(doc_id, n_results=5)

    # Fallback to regular search
    return get_vector_store().search(message, n_results=5)


# Run with: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    import socket

    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    # Try port 8000 first, fall back to 8001 if busy
    port = 8000
    if is_port_in_use(port):
        port = 8001
        print(f"Port 8000 in use, using port {port}")

    print(f"Starting Disaster Clippy on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
