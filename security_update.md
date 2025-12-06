# Security Update Plan

## Overview

This document outlines the security architecture for separating public Railway deployment from local admin functionality, ensuring API keys are never exposed in the GitHub repository.

## Access Levels

| Feature | Railway (Public) | Local Admin | Global Admin |
|---------|-----------------|-------------|--------------|
| Chat UI | Yes | Yes | Yes |
| Pinecone Read (search) | Yes | Yes | Yes |
| Admin UI (`/useradmin/`) | No | Yes | Yes |
| R2 Read (download backups) | No | Yes | Yes |
| R2 Write (submissions) | No | Yes | Yes |
| Pinecone Write (curate index) | No | No | Yes |
| R2 Full (official backups) | No | No | Yes |

---

## Audit Results

### R2 Direct Access (requires R2_* env vars)

| File | Function | Purpose | Who needs it |
|------|----------|---------|--------------|
| `admin/routes/packs.py:160` | `get_cloud_sources()` | List available backups | Local Admin |
| `admin/routes/packs.py:494` | `download_pack_from_r2()` | Download backup files | Local Admin |
| `admin/cloud_upload.py:252` | `upload_backup()` | Upload to backups/ | Global Admin only |
| `admin/cloud_upload.py:591` | `upload_source_backup()` | Upload source | Global Admin only |
| `admin/cloud_upload.py:729` | `get_cloud_backups()` | List backups | Both |

### Pinecone Direct Access (requires PINECONE_API_KEY)

| File | Function | Purpose | Who needs it |
|------|----------|---------|--------------|
| `app.py:126` | Search via `get_vector_store()` | Online search | Railway + Local Admin |
| `admin/cloud_upload.py:848` | `pinecone_push()` | Write to index | Global Admin only |
| `admin/cloud_upload.py:928` | `pinecone_delete_namespace()` | Delete vectors | Global Admin only |
| `admin/cloud_upload.py:793` | `pinecone_compare()` | Compare local vs cloud | Global Admin only |

---

## Update Plan

### Phase 1: Create Public Proxy Endpoints (Railway)

Add to `app.py` (not admin router - these are public):

```
GET  /api/cloud/sources          - List available backups (proxies R2)
GET  /api/cloud/download/{id}    - Stream backup download (proxies R2)
POST /api/cloud/submit           - Submit content (writes to R2 submissions/)
```

These endpoints:
- Don't require API keys from caller
- Railway has the real R2/Pinecone keys
- Can add rate limiting later

### Phase 2: Update Local Admin Code

Modify local admin to detect "proxy mode":
- If `R2_ACCESS_KEY_ID` is set -> use direct R2 (global admin)
- If not set -> call Railway proxy endpoints

Files to update:
- `admin/routes/packs.py` - `get_cloud_sources()`, `download_pack_from_r2()`
- Add `RAILWAY_PROXY_URL` env var for local admins

### Phase 3: Pinecone Search (Already OK!)

The chat endpoints (`/chat`, `/api/v1/chat`) already work as proxies:
- Railway has `PINECONE_API_KEY`
- Local admins call Railway's chat endpoint
- Search works without local admins having the key

Only change needed: Ensure local admin uses Railway URL for chat when online.

### Phase 4: Protect Write Operations

Already done via `require_global_admin()`:
- `pinecone_push` - requires `ADMIN_MODE=global`
- `pinecone_delete_namespace` - requires `ADMIN_MODE=global`
- R2 uploads - only work with full R2 keys

---

## Environment Variables Summary

| Deployment | Keys Needed | Notes |
|------------|-------------|-------|
| **Railway (Public)** | `PINECONE_API_KEY`, `R2_*`, `PUBLIC_MODE=true` | Full keys, no admin UI |
| **Local Admin** | `RAILWAY_PROXY_URL` only | Calls Railway for cloud access |
| **Global Admin** | All keys + `ADMIN_MODE=global` | Full access |

---

## Current Implementation Status

### Completed

- [x] `PUBLIC_MODE=true` blocks admin UI on Railway (`admin/app.py`)
- [x] `ADMIN_MODE=global/local` controls global features
- [x] "Local Settings" link removed from public chat page
- [x] Pinecone write operations require global admin

### TODO

- [x] Phase 1: Create `/api/cloud/*` proxy endpoints (`app.py`)
- [x] Add rate limiting to proxy endpoints (slowapi)
- [x] Phase 2: Update local admin to use proxy when no R2 keys (`admin/cloud_proxy.py`, `admin/routes/packs.py`)
- [x] Phase 3: Configure local admin to use Railway for online search
  - Added `chat()` and `search()` methods to `CloudProxyClient` (`admin/cloud_proxy.py`)
  - Added `should_use_proxy_for_search()` to `LocalConfig` (`admin/local_config.py`)
  - Updated `AIService.search()` to use proxy when configured (`admin/ai_service.py`)
- [ ] Test full flow: Local admin -> Railway proxy -> R2/Pinecone

---

## Architecture Diagram

```
                    RAILWAY (Public)
                    ================
                    - PUBLIC_MODE=true
                    - Has all API keys
                    - Chat UI only
                    - Proxy endpoints for cloud access
                           |
                           | /api/cloud/* endpoints
                           | /api/v1/chat endpoint
                           v
    +------------------+       +------------------+
    |   LOCAL ADMIN    |       |   GLOBAL ADMIN   |
    +------------------+       +------------------+
    - RAILWAY_PROXY_URL        - All API keys
    - Own ChromaDB             - ADMIN_MODE=global
    - Own backup folder        - Direct R2/Pinecone
    - Own Ollama               - Full write access
    - Calls Railway for
      cloud operations
```

---

## R2 Bucket Structure (Two Buckets)

For maximum security, use TWO separate R2 buckets with scoped API tokens:

```
disaster-clippy-backups/        # BUCKET 1: Official content
  backups/
    _master.json                # Source registry
    appropedia/
    practical-action/
    ...

disaster-clippy-submissions/    # BUCKET 2: User submissions
  pending/                      # Awaiting review
  approved/                     # Approved (copied to backups bucket)
  rejected/                     # Rejected submissions
```

### API Token Permissions

| Token | Backups Bucket | Submissions Bucket |
|-------|----------------|-------------------|
| **Railway** | Read only | Write only |
| **Global Admin** | Read + Write | Read + Delete |

### Environment Variables

```
# Railway (limited access)
R2_BACKUPS_BUCKET=disaster-clippy-backups
R2_SUBMISSIONS_BUCKET=disaster-clippy-submissions

# Global Admin (full access) - same vars, different token with more permissions
```

### Server-Side Copy

When global admin approves a submission, it's copied directly within R2:
- No network transfer needed (data stays in Cloudflare)
- Uses `approve_submission()` from `offline_tools/cloud/r2.py`

---

## Notes

- No API keys in GitHub repository
- Local admins get full independence through local sources/ChromaDB/Ollama
- Cloud access (backups, online search) goes through Railway proxy
- Global admin is the only one with direct cloud write access
- Approved submissions are server-side copied between buckets (efficient)
