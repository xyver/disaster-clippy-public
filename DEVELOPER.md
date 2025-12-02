# Disaster Clippy - Developer Documentation

Welcome to the Disaster Clippy developer docs. Choose the guide that matches your role:

---

## Repository Structure (Public vs Private)

This project uses a two-layer architecture. The **public GitHub repository** contains everything needed for local users, while **private components** stay on the maintainer's machine.

### PUBLIC GITHUB (What Users Get)

```
disaster-clippy/
|-- app.py                    # FastAPI chat interface
|-- local_cli.py              # CLI for local admins
|-- useradmin/                # Local admin panel (/useradmin/)
|-- sourcepacks/              # Pack tools (shared utilities)
|   |-- pack_tools.py         # Metadata/indexing functions
|-- vectordb/                 # ChromaDB local storage
|-- offline_tools/            # Tools to index local backups
|-- storage/                  # R2 client (limited access)
|-- config/
|   |-- sources.json          # Official sources registry
|-- templates/                # Web UI templates
|-- static/                   # CSS/JS assets
|-- DEVELOPER-LOCAL.md        # User documentation
```

### PRIVATE (Maintainer Only - Not in Public Repo)

```
disaster-clippy/
|-- admin/                    # Streamlit admin dashboard
|-- scraper/                  # Web scrapers (rate-limited, API access)
|-- ingest.py                 # CLI for scraping content
|-- sync.py                   # Pinecone sync tools
|-- .env                      # API keys (Pinecone write, full R2)
|-- DEVELOPER-PARENT.md       # Maintainer documentation
```

### Access Levels

| Role | Vector DB | R2 Storage | What They Use |
|------|-----------|------------|---------------|
| End User | ChromaDB (local) | Read backups/ only | useradmin panel |
| Local Admin | ChromaDB (local) | Write submissions/, Read backups/ | useradmin + pack tools |
| Global Admin | Pinecone (write) | Full access | Streamlit admin + scrapers |

---

## For End Users (Local System)

**[DEVELOPER-LOCAL.md](DEVELOPER-LOCAL.md)** - Setting up your own offline system

If you want to:
- Run Disaster Clippy on your own computer
- Add personal sources and PDFs
- Set up offline backups (ZIM, HTML, PDF)
- Configure connection modes (online/hybrid/offline)
- Use the Local Admin Panel at `/useradmin/`

---

## For Maintainers (Parent System)

**[DEVELOPER-PARENT.md](DEVELOPER-PARENT.md)** - Managing the global infrastructure

If you want to:
- Manage the central Pinecone database
- Deploy to Railway
- Add and curate official sources
- Run the Streamlit admin dashboard
- Sync local changes to production
- Create new scrapers

---

## Quick Reference

| Task | Guide | Repo | Key Command/URL |
|------|-------|------|-----------------|
| Run chat locally | Local | Public | `python app.py` -> localhost:8000 |
| Configure backups | Local | Public | localhost:8000/useradmin/ |
| Index local backups | Local | Public | `python local_cli.py index-html ...` |
| Submit source pack | Local | Public | useradmin -> Cloud Upload |
| Manage global sources | Parent | Private | `streamlit run admin/app.py` |
| Add scraped content | Parent | Private | `python ingest.py scrape ...` |
| Sync to Pinecone | Parent | Private | `python sync.py --remote pinecone push` |

---

## Other Documentation

- [SUMMARY.md](SUMMARY.md) - Executive summary (non-technical overview)
- [README.md](README.md) - Project overview
- [CONTEXT.md](CONTEXT.md) - Design decisions and rationale
- [ROADMAP.md](ROADMAP.md) - Future plans

---

*Last Updated: December 2025*
