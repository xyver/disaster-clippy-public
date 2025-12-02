# Disaster Clippy

**Evidence-based emergency preparedness guidance through conversational AI**

Disaster Clippy helps you find actionable information from trusted sources - educational guides, DIY projects, government reports, and research papers. Ask questions in plain language and get recommendations with source attribution.

---

## What Can I Ask?

- "How do I purify water in an emergency?"
- "What's the best way to build a solar cooker?"
- "Show me guides on food preservation"
- "How do I prepare for a wildfire?"

The system searches 800+ documents from trusted sources and provides answers with links to the original content.

---

## Quick Start

### Option 1: Use the Hosted Version

Coming soon - a publicly accessible version.

### Option 2: Run Locally

```bash
# Clone the repository
git clone <repo-url>
cd disaster-clippy

# Install dependencies
pip install -r requirements.txt

# Configure (add your API keys)
cp .env.example .env
# Edit .env with your OpenAI API key

# Start the chat interface
python app.py
# Visit http://localhost:8000
```

### Option 3: Use the API

Embed Disaster Clippy on your website:

```bash
curl -X POST "https://your-server/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I filter water?"}'
```

See the [API documentation](#api-endpoints) below.

---

## Content Sources

| Source | Documents | Topics |
|--------|-----------|--------|
| Appropedia | 150 | Appropriate technology, water, sanitation, shelter |
| BuildItSolar | 337 | DIY solar projects, heating, cooling |
| SolarCooking Wiki | 176 | Solar cooking, food safety |
| PDF Guides | 61+ | Pandemic preparedness, emergency management |
| Bitcoin Docs | 100 | Cryptocurrency reference |

**Total: 824+ documents from verified, openly-licensed sources**

All content is Creative Commons or Public Domain with full attribution.

---

## Features

### For Users

- **Natural language search** - Ask questions like you would ask a person
- **Source attribution** - Every answer links to the original source
- **Document types** - Results tagged as Guide, Article, Research, or Product
- **Multi-turn conversation** - Follow-up questions keep context

### For Organizations

- **External API** - Embed the chat on your own website
- **Custom databases** - Create your own vector database from selected sources
- **Offline support** - (Coming soon) Run without internet access
- **Location-aware** - (Coming soon) Prioritize locally-relevant content

---

## API Endpoints

### Chat API

**POST /api/v1/chat**

```json
// Request
{"message": "How do I build a solar oven?", "session_id": "optional"}

// Response
{"response": "Here are several approaches...", "session_id": "abc123"}
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface (web page) |
| `/health` | GET | Health check |
| `/stats` | GET | System statistics |
| `/api/v1/embed` | GET | Get embed code for websites |

---

## Documentation

| Document | Purpose |
|----------|---------|
| [DEVELOPER.md](DEVELOPER.md) | Setup, configuration, deployment, technical architecture |
| [ROADMAP.md](ROADMAP.md) | Future plans and feature development |
| [CONTEXT.md](CONTEXT.md) | Complete project context for AI assistants |

---

## Tech Stack

- **Backend**: Python, FastAPI
- **AI**: OpenAI GPT-4o-mini or Claude (configurable)
- **Embeddings**: OpenAI API or local sentence-transformers
- **Vector Database**: ChromaDB (local) or Pinecone (cloud)
- **Admin UI**: Streamlit
- **Scrapers**: BeautifulSoup, MediaWiki API, PyMuPDF

---

## License

MIT

Content sources retain their original licenses (CC-BY, CC-BY-SA, Public Domain).

---

## Contributing

See [DEVELOPER.md](DEVELOPER.md) for development setup.

Key areas:
- Add new content scrapers
- Improve search quality
- Build location-aware features
- Test and documentation

---

*Version 0.8 - December 2025*
