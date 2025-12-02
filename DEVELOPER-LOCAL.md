# Disaster Clippy - Local User Guide

How to set up your own offline disaster preparedness assistant.

This guide is for **end users** who want to run their own local Disaster Clippy system with personal sources and offline backup capability. This is what you get from the **public GitHub repository**.

For global database maintainers (private tools), see [DEVELOPER-PARENT.md](DEVELOPER-PARENT.md).

---

## What's Included (Public Repo)

When you clone the public repository, you get:
- Chat interface and API
- Local Admin Panel (/useradmin/)
- ChromaDB for local vector storage
- Backup indexers (for ZIM, HTML, PDF files you already have)
- Cloud submission tools (submit sources for review)
- Shared pack tools (read-only, for generating metadata)

What's NOT included (maintainer-only):
- Web scrapers (to prevent abuse)
- Streamlit admin dashboard
- Pinecone write access
- ingest.py CLI

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Local Admin Panel](#local-admin-panel)
4. [Setting Up Offline Backups](#setting-up-offline-backups)
5. [Adding Your Own Sources](#adding-your-own-sources)
6. [Connection Modes](#connection-modes)
7. [ZIM Files](#zim-files)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What You Get

When you run Disaster Clippy locally, you have:

1. **Chat Interface** - Ask questions, get AI-powered answers from your knowledge base
2. **Local Admin Panel** - Configure backup paths, connection modes, and settings
3. **Offline Capability** - Access content even without internet (with proper setup)
4. **Personal Sources** - Add your own documents, PDFs, and websites

### System Architecture

```
+------------------+     +-------------------+
|   Chat UI        | --> | Local Vector DB   |
| localhost:8000   |     |   (ChromaDB)      |
+------------------+     +-------------------+
        |                         |
        v                         v
+------------------+     +-------------------+
| Local Admin      |     | Backup Files      |
| /useradmin/      |     | ZIM, HTML, PDF    |
+------------------+     +-------------------+
```

### Requirements

- Python 3.9+
- 4GB RAM minimum (8GB recommended for local AI)
- 1GB disk space minimum (more for backups)
- Internet connection (for initial setup and online mode)

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/disaster-clippy.git
cd disaster-clippy
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```bash
# Minimum required - for embeddings
OPENAI_API_KEY=sk-proj-...

# Optional - use Claude for chat instead of GPT
# ANTHROPIC_API_KEY=sk-ant-...
# LLM_PROVIDER=anthropic

# Use local database (default for personal use)
VECTOR_DB_MODE=local

# Optional - use free local embeddings (no API cost)
# EMBEDDING_MODE=local
```

### 3. Start the Application

```bash
python app.py
```

Open your browser:
- **Chat Interface**: http://localhost:8000
- **Local Admin**: http://localhost:8000/useradmin/

### 4. Configure Your System

1. Go to the Local Admin Panel
2. Set your backup folder paths (ZIM files, HTML backups, PDFs)
3. Choose your connection mode (Hybrid recommended)
4. Save settings

---

## Local Admin Panel

Access at: `http://localhost:8000/useradmin/`

### Status Bar

Shows:
- Internet connectivity (Connected/Disconnected)
- Current connection mode
- Number of backup files detected

### Connection Mode

Choose how your system handles internet:

| Mode | Description | When to Use |
|------|-------------|-------------|
| **Online Only** | Always uses internet for queries | When you have reliable internet |
| **Hybrid** (Recommended) | Uses internet when available, falls back to offline | Best of both worlds |
| **Offline Only** | Never connects to internet | Air-gapped systems, no internet |

### Backup Locations

Configure where your offline backups are stored:

- **ZIM Folder**: Path to folder containing `.zim` files (Wikipedia, Wikihow, etc.)
- **HTML Folder**: Path to folder containing HTML page backups
- **PDF Folder**: Path to folder containing PDF documents

Example:
```
ZIM Folder: C:\Backups\ZIM
HTML Folder: C:\Backups\HTML
PDF Folder: C:\Backups\PDF
```

Click "Verify" after entering a path to confirm it exists and is readable.

### Settings File

Your settings are saved to: `config/local_settings.json`

```json
{
  "backup_paths": {
    "zim_folder": "C:\\Backups\\ZIM",
    "html_folder": "C:\\Backups\\HTML",
    "pdf_folder": "C:\\Backups\\PDF"
  },
  "offline_mode": "hybrid",
  "auto_fallback": true,
  "cache_responses": true
}
```

---

## Setting Up Offline Backups

### Folder Structure

Organize your backups like this:

```
C:\Backups\
|-- ZIM\
|   |-- wikipedia_en_medicine.zim
|   |-- wikihow_en_all.zim
|   |-- appropedia.zim
|
|-- HTML\
|   |-- builditsolar\
|   |   |-- solar_cooker_1.html
|   |   |-- water_heater.html
|   |
|   |-- mysite\
|       |-- page1.html
|
|-- PDF\
    |-- emergency_manual.pdf
    |-- first_aid_guide.pdf
```

### Getting ZIM Files

ZIM files are compressed offline archives. Download from:

1. **Kiwix Library**: https://library.kiwix.org/
   - Wikipedia (by topic: medicine, technology, etc.)
   - Wikihow
   - StackExchange sites
   - And more

2. **Direct Downloads**:
   - Search for `[topic] kiwix zim download`
   - Files range from 50MB to several GB

Recommended starter ZIMs:
- `wikipedia_en_medicine` (~500MB) - Medical reference
- `wikihow_en_all` (~2GB) - How-to guides
- `wikibooks_en_all` (~1GB) - Technical books

### Creating HTML Backups

You can create HTML backups in several ways:

**Option 1: Browser Save**
- Visit any page and use "Save As" -> "Webpage, Complete"
- Organize saved pages into folders by source

**Option 2: HTTrack / wget**
- Use [HTTrack](https://www.httrack.com/) to mirror websites
- Or use wget: `wget -r -l 2 -p https://example.com`

**Option 3: Kiwix ZIM Files**
- Download pre-made ZIM archives from [library.kiwix.org](https://library.kiwix.org)
- Already compressed and indexed

Once you have backups, configure the paths in useradmin and use the indexing tools.

---

## Adding Your Own Sources

### From the Chat Interface

1. Click "Select Sources" in the stats bar
2. Check/uncheck sources to include in searches
3. Preferences are saved to your browser

### Indexing Local Backups (via Admin Panel)

The Local Admin Panel at `/useradmin/` lets you index backup files you already have:

1. Go to **Packs** tab in useradmin
2. Point to your backup folder (HTML or ZIM files)
3. Click **Generate Metadata** to scan the content
4. Click **Index to ChromaDB** to make it searchable

This works with:
- **HTML folders** - Downloaded websites, saved pages
- **ZIM files** - Kiwix offline archives
- **PDF folders** - Document collections

### Submitting Sources for Official Review

If you have a great source with proper licensing:

1. Prepare your backup (HTML folder or ZIM file)
2. Go to **Cloud Upload** in useradmin
3. The system checks: config + metadata + backup + verified license
4. Click **Submit for Review**
5. Files go to `submissions/` folder in cloud storage
6. Global admin reviews and approves/denies

Note: You submit DATA (the backup + metadata), not code. The admin regenerates all indexes server-side.

### Using the CLI (Power Users)

For command-line access, use `local_cli.py`:

```bash
# Generate metadata from HTML backup
python local_cli.py metadata --path ./backups/mysite --output metadata.json

# Index HTML backup to local ChromaDB
python local_cli.py index-html --path ./backups/mysite --source-id mysite

# Index ZIM file to local ChromaDB
python local_cli.py index-zim --path ./backups/wikipedia.zim --source-id wikipedia

# Check if pack is ready for submission
python local_cli.py check --source-id mysite

# Create manifest for submission
python local_cli.py manifest --source-id mysite --backup-path ./backups/mysite
```

Run `python local_cli.py --help` for full usage.

---

## Connection Modes

### Online Only

- All queries go to the cloud database
- Requires constant internet
- Best search quality
- Lowest local storage

### Hybrid (Recommended)

- Uses internet when available
- Falls back to local data when offline
- Best balance of quality and reliability
- Caches responses for future offline use

### Offline Only

- Never connects to internet
- Uses only local ChromaDB + backup files
- Requires pre-indexed content
- Works in air-gapped environments

### Auto-Fallback

When enabled (default), the system automatically:
1. Checks internet connectivity
2. If unavailable, switches to offline mode
3. Shows "offline" indicator in the UI

---

## ZIM Files

### What Are ZIM Files?

ZIM (Zeno IMproved) is a file format for storing web content offline. Created by Kiwix for offline Wikipedia access.

Benefits:
- Highly compressed (10:1 or better)
- Full-text search built-in
- Works completely offline
- Includes images and formatting
- Standard format with multiple readers

### Using ZIM Files

**Option 1: Kiwix Reader**

Download Kiwix from https://kiwix.org and open ZIM files directly.

**Option 2: Integrated Reader (Coming Soon)**

Future versions will include a built-in ZIM reader at `/zim/[source]/[path]`.

### Finding ZIM Content

When you search and get results from ZIM-indexed content:
- Articles show an "offline" badge
- Titles are displayed but not clickable
- Content is searchable but browsing requires Kiwix

---

## Troubleshooting

### "No sources indexed yet"

Your database is empty. Index some content using the Local Admin panel:
1. Go to `/useradmin/` -> Packs tab
2. Point to a backup folder with HTML or ZIM files
3. Click "Generate Metadata" then "Index to ChromaDB"

### "Unable to connect to OpenAI"

Check your API key in `.env`:
```bash
OPENAI_API_KEY=sk-proj-...
```

Or switch to local embeddings (free):
```bash
EMBEDDING_MODE=local
```

### "Port 8000 already in use"

The system automatically tries port 8001. Or kill the process using port 8000:

```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# Mac/Linux
lsof -i :8000
kill -9 <pid>
```

### Backup folder not detected

1. Verify the path exists
2. Check you have read permissions
3. Use full absolute paths (e.g., `C:\Backups\ZIM` not `.\Backups\ZIM`)

### Offline mode not working

1. Ensure you have content indexed locally (check stats in useradmin)
2. Check backup paths are configured in Local Admin
3. Verify ZIM files are valid (try opening in Kiwix)

### Slow performance

- Use local embeddings: `EMBEDDING_MODE=local`
- Reduce `n_results` in searches
- Index only the sources you need

---

## File Locations

| Item | Location |
|------|----------|
| Settings | `config/local_settings.json` |
| Local Database | `data/chroma/` |
| Metadata Index | `data/metadata/` |
| Environment | `.env` |

---

## Next Steps

1. **Set up backup folders** - Configure paths in Local Admin
2. **Download ZIM files** - Get offline content from Kiwix
3. **Index personal sources** - Add your own PDFs and websites
4. **Test offline mode** - Disconnect internet and verify it works
5. **Export your setup** - Share with others or transfer to another machine

---

## Getting Help

- **Project Issues**: https://github.com/yourusername/disaster-clippy/issues
- **Documentation**: See other docs in the project root
- **Local Admin**: Check `/useradmin/` for system status

---

*Last Updated: December 2025*
*For Local Users - Version 1.0*
