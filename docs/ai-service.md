# AI Service

This document covers the AI search and response system, including connection modes, search diversity, and content filtering.

---

## Table of Contents

1. [AI Service Architecture](#ai-service-architecture)
2. [Connection Modes](#connection-modes)
3. [Connection Manager](#connection-manager)
4. [API Endpoints](#api-endpoints)
5. [Source Filtering](#source-filtering)
6. [Search Result Diversity](#search-result-diversity)
7. [Content Filtering](#content-filtering)
8. [Chat Link Behavior](#chat-link-behavior)

---

## AI Service Architecture

The AI service (`admin/ai_service.py`) provides a unified interface for search and response generation across all connection modes.

### Using the AI Service

```python
from admin.ai_service import get_ai_service

# Get the singleton service
ai = get_ai_service()

# Search (automatically uses correct method based on mode)
result = ai.search("how to filter water", n_results=10)
print(f"Found {len(result.articles)} articles via {result.method}")

# Generate response
response = ai.generate_response(query, context, history)
print(f"Response via {response.method}: {response.text}")

# Streaming response
for chunk in ai.generate_response_stream(query, context, history):
    print(chunk, end="", flush=True)
```

---

## Connection Modes

| Mode | Search | Response | Pinging |
|------|--------|----------|---------|
| `online_only` | Semantic (embedding API) | Cloud LLM | Yes (warn on disconnect) |
| `hybrid` | Semantic with keyword fallback | Cloud LLM with Ollama fallback | Yes (detect recovery) |
| `offline_only` | Keyword only | Ollama or simple response | No |

### When to Use Each Mode

| Mode | Description | When to Use |
|------|-------------|-------------|
| **Online Only** | Always uses internet for queries | When you have reliable internet |
| **Hybrid** (Recommended) | Uses internet when available, falls back to offline | Best of both worlds |
| **Offline Only** | Never connects to internet | Air-gapped systems, no internet |

---

## Connection Manager

The connection manager (`admin/connection_manager.py`) handles smart connectivity detection:

```python
from admin.connection_manager import get_connection_manager

conn = get_connection_manager()

# Check if online
if conn.should_try_online():
    # Try online API

# Report success/failure
conn.on_api_success()  # Resets ping timer
conn.on_api_failure()  # Triggers immediate ping check

# Get status for frontend
status = conn.get_status()
# Returns: {mode, is_online, temporarily_offline, effective_mode, ...}
```

### Connection States

| State | Color | Description |
|-------|-------|-------------|
| `online` | green | Securely connected, recent successful API call |
| `checking` | blue | Currently verifying connection |
| `unstable` | yellow | Hybrid mode with intermittent issues |
| `disconnected` | red | Online mode but connection lost |
| `offline` | gray | User intentionally in offline mode |
| `recovering` | blue | Was offline, now detecting recovery |

### Connection Status Response

The `/api/v1/connection-status` endpoint returns unified status data:

```json
{
  "mode": "hybrid",
  "state": "online",
  "state_label": "Online",
  "state_color": "green",
  "state_icon": "check",
  "message": "Connected to cloud services",
  "is_online": true,
  "temporarily_offline": false,
  "effective_mode": "hybrid_online"
}
```

### UI Implementations

- **Dashboard** (`admin/templates/dashboard.html`): Shows "Connection State" card with label, color, and message
- **Settings** (`admin/templates/settings.html`): Status bar with colored dot and state text
- **Chat** (`templates/index.html` + `static/chat.js`): Header indicator with colored dot, auto-refreshes every 30 seconds

All three pages fetch from the same `/api/v1/connection-status` endpoint for consistency.

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/chat` | POST | Standard chat (waits for full response) |
| `/api/v1/chat/stream` | POST | Streaming chat (SSE) |
| `/api/v1/connection-status` | GET | Get current connection status |
| `/api/v1/ping` | POST | Trigger connectivity check |
| `/sources` | GET | List available sources with document counts |
| `/welcome` | GET | Get welcome message and stats |

---

## Source Filtering

Users can filter search results by source using the "Select Sources" dropdown in the chat interface:

- **Select All**: Search all indexed sources (default)
- **Select None**: Disable search (useful for testing)
- **Individual sources**: Check/uncheck specific sources to include

The selection is persisted to localStorage (`clippy_selected_sources`) so it survives page refreshes.

**API Usage:**
```javascript
// Filter to specific sources
fetch('/api/v1/chat/stream', {
    method: 'POST',
    body: JSON.stringify({
        message: "how to filter water",
        sources: ["appropedia", "wikihow"]  // Only search these sources
    })
});
```

---

## Search Result Diversity

Search results are re-ranked to ensure diversity across sources, preventing any single source from dominating results.

### How It Works

1. **Search Phase**: Retrieves 15 candidate results from vector DB
2. **Doc Type Prioritization**: Boosts guides over articles (configurable)
3. **Source Diversity**: Limits to 2 results per source, then backfills
4. **Final Output**: Returns top 5 diverse results to user

### Implementation

```python
# app.py - ensure_source_diversity()
def ensure_source_diversity(articles, max_per_source=2, total_results=5):
    # Group by source
    # Round-robin: take up to max_per_source from each source
    # Backfill remaining slots with highest-scored unused articles
```

**Example behavior:**
- Input: `[wiki1, wiki2, wiki3, wiki4, ready1, ready2, appro1]`
- Output: `[wiki1, ready1, appro1, wiki2, ready2]`

### User Override

Users can still filter to a single source using the source filter in chat UI. When only one source is selected, all 5 results come from that source.

---

## Content Filtering

### Minimum Content Length

Articles with less than 100 characters of extracted text are filtered out during metadata generation. This removes stub pages, redirects, and other low-value content.

**Configured in:**
- `source_manager.py` - Filter during ZIM metadata generation
- `zim_utils.py` - Default parameter for ZIM inspection

**Previous value:** 50 characters
**Current value:** 100 characters

---

## Chat Link Behavior

Links in chat responses and the articles sidebar open in new tabs to preserve chat history:

- **ZIM article links**: Open in new tab with `target="_blank"`
- **External URLs**: Open in new tab with `rel="noopener noreferrer"`
- **Markdown links**: Parsed and converted to clickable links that open in new tabs

This prevents users from losing their conversation when clicking to read an article.

---

## Related Documentation

- [Architecture](architecture.md) - System design and connection modes
- [Language Packs](language-packs.md) - Translation integration with AI service

---

*Last Updated: December 2025*
