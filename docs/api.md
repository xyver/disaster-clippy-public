# External API Reference

API documentation for embedding Disaster Clippy on external websites.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Endpoints](#endpoints)
3. [Source Filtering](#source-filtering)
4. [Embed Widget](#embed-widget)
5. [Rate Limits](#rate-limits)
6. [Response Formats](#response-formats)

---

## Quick Start

Send a question, get an AI response with citations:

```bash
curl -X POST "https://disaster-clippy.up.railway.app/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I purify water?"}'
```

Response:
```json
{
  "response": "There are several effective methods to purify water...",
  "session_id": "2025-01-15T10:30:00"
}
```

---

## Endpoints

### POST /api/v1/chat

Standard chat endpoint. Waits for complete response.

**Request:**
```json
{
  "message": "How do I filter water?",
  "session_id": "optional-session-id",
  "source_mode": "all",
  "exclude": ["bitcoin"],
  "include": []
}
```

**Response:**
```json
{
  "response": "Here are several approaches to filtering water...",
  "session_id": "abc123"
}
```

### POST /api/v1/chat/stream

Streaming chat endpoint. Returns Server-Sent Events (SSE) for real-time response.

**Request:** Same as `/api/v1/chat`

**Response:** SSE stream with chunks:
```
data: {"chunk": "Here are "}
data: {"chunk": "several "}
data: {"chunk": "approaches..."}
data: {"done": true, "session_id": "abc123"}
```

### GET /api/v1/sources

List available sources for filtering. Use these IDs in `include`/`exclude` fields.

**Response:**
```json
{
  "sources": [
    {"id": "appropedia", "name": "appropedia", "count": 13522},
    {"id": "bitcoin", "name": "bitcoin", "count": 1812},
    {"id": "builditsolar2", "name": "Builditsolar2", "count": 876},
    {"id": "ready_gov_site", "name": "ready_gov_site", "count": 263},
    {"id": "wikipedia-medical", "name": "wikipedia-medical", "count": 70762},
    {"id": "wikipedia_climate_change", "name": "wikipedia_climate_change", "count": 5269}
  ],
  "total_documents": 92504
}
```

### GET /sources

Detailed source information (used by admin panel). Includes vector availability flags.

### GET /api/v1/connection-status

Check API connection state.

---

## Source Filtering

Control which knowledge sources are searched. Source IDs match those shown in the chat interface and returned by `/api/v1/sources`.

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source_mode` | string | `"all"` | `"all"` = search all sources, `"none"` = start with no sources |
| `exclude` | array | `[]` | Source IDs to exclude (when mode="all") |
| `include` | array | `[]` | Source IDs to include (when mode="none") |
| `sources` | array | `null` | Legacy: explicit list of source IDs (overrides mode) |

### Behavior Matrix

| source_mode | exclude | include | Result |
|-------------|---------|---------|--------|
| null/omitted | null | null | Search ALL sources (default) |
| "all" | null | null | Search ALL sources |
| "all" | ["bitcoin"] | null | All sources EXCEPT bitcoin |
| "none" | null | ["appropedia"] | ONLY appropedia |
| "none" | null | null | No sources (returns helpful message) |

**Precedence:** If the legacy `sources` field is provided, it takes precedence over `source_mode`/`include`/`exclude` for backwards compatibility.

### Examples

**Search all sources (default):**
```json
{"message": "How to build a solar heater?"}
```

**Search all except specific sources:**
```json
{
  "message": "First aid for burns",
  "source_mode": "all",
  "exclude": ["bitcoin", "builditsolar2"]
}
```

**Search only specific sources:**
```json
{
  "message": "Medical emergency procedures",
  "source_mode": "none",
  "include": ["wikipedia-medical", "ready_gov_site"]
}
```

**Legacy format (still supported):**
```json
{
  "message": "Solar panel basics",
  "sources": ["appropedia", "builditsolar2"]
}
```

### Error Handling

The API is forgiving - it uses fuzzy matching to suggest corrections and falls back to searching all sources when needed. Warnings are prepended to the response text.

**Typo in source ID (fuzzy matching):**

The API uses fuzzy matching to suggest the correct source ID, then falls back to all sources:

```json
// Request with typo
{"message": "water purification", "source_mode": "none", "include": ["apropedia"]}

// Response - suggests correction, searches all sources
{
  "response": "Note: Source 'apropedia' not found. Did you mean 'appropedia'?\nSearching all sources instead.\n\nThere are several effective methods..."
}
```

**Multiple typos:**
```json
// Request with multiple typos
{"message": "water", "source_mode": "none", "include": ["bittcoin", "reddy_gov"]}

// Response - each typo gets a suggestion
{
  "response": "Note: Source 'bittcoin' not found. Did you mean 'bitcoin'?\nNote: Source 'reddy_gov' not found. Did you mean 'ready_gov_site'?\nSearching all sources instead.\n\nHere's what I found..."
}
```

**Contradictory include/exclude:**
```json
// Request - bitcoin both excluded and included
{"message": "solar", "source_mode": "all", "exclude": ["bitcoin"], "include": ["bitcoin"]}

// Response - include wins, with warning
{
  "response": "Note: 'bitcoin' was both excluded and included - including it.\n\nHere's information about solar..."
}
```

**Empty include list (intentional):**
```json
// Request - explicitly no sources
{"message": "Hello", "source_mode": "none", "include": []}

// Response - helpful message
{
  "response": "No sources selected to search. You can remove source filters to search all available knowledge, or check your include/exclude lists."
}
```

---

## Embed Widget

### Ready-to-Use Widget

Visit `/static/embed-widget.html` for a complete example with HTML, CSS, and JavaScript.

### Basic Implementation

```html
<div id="clippy-chat">
  <div id="clippy-messages"></div>
  <input type="text" id="clippy-input" placeholder="Ask a question...">
  <button onclick="sendMessage()">Send</button>
</div>

<script>
const API_URL = "https://disaster-clippy.up.railway.app/api/v1/chat";
let sessionId = null;

async function sendMessage() {
  const input = document.getElementById("clippy-input");
  const message = input.value.trim();
  if (!message) return;

  // Show user message
  addMessage(message, "user");
  input.value = "";

  // Send to API
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      message: message,
      session_id: sessionId
    })
  });

  const data = await response.json();
  sessionId = data.session_id;

  // Show AI response
  addMessage(data.response, "bot");
}

function addMessage(text, type) {
  const messages = document.getElementById("clippy-messages");
  const div = document.createElement("div");
  div.className = `message ${type}`;
  div.textContent = text;
  messages.appendChild(div);
}
</script>
```

### With Source Filtering

```javascript
// Only search medical sources
const response = await fetch(API_URL, {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({
    message: message,
    session_id: sessionId,
    source_mode: "none",
    include: ["wikipedia-medical", "ready_gov_site"]
  })
});
```

### Streaming Implementation

```javascript
async function sendMessageStreaming() {
  const input = document.getElementById("clippy-input");
  const message = input.value.trim();

  const response = await fetch(API_URL + "/stream", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({message, session_id: sessionId})
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let responseText = "";

  while (true) {
    const {done, value} = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    // Parse SSE format
    const lines = chunk.split("\n");
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const data = JSON.parse(line.slice(6));
        if (data.chunk) {
          responseText += data.chunk;
          updateMessage(responseText);
        }
        if (data.session_id) {
          sessionId = data.session_id;
        }
      }
    }
  }
}
```

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/api/v1/chat` | 10 requests/minute per IP |
| `/api/v1/chat/stream` | 10 requests/minute per IP |
| `/api/v1/sources` | No limit (cached) |

Rate limit exceeded returns HTTP 429.

---

## Response Formats

### Success Response

```json
{
  "response": "AI-generated response text with citations...",
  "session_id": "unique-session-identifier"
}
```

### Error Response

```json
{
  "detail": "Error description"
}
```

### Common HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid JSON) |
| 429 | Rate limit exceeded |
| 500 | Server error |

---

## Session Management

Sessions maintain conversation context for follow-up questions.

- **New session:** Omit `session_id` or send `null`
- **Continue session:** Include the `session_id` from previous response
- **Session lifetime:** Sessions expire after 30 minutes of inactivity

Example conversation:
```javascript
// First message - no session
let response = await chat("What is solar power?");
let sessionId = response.session_id;

// Follow-up - include session
response = await chat("How efficient is it?", sessionId);
// AI understands "it" refers to solar power
```

---

*Last Updated: December 2025*
