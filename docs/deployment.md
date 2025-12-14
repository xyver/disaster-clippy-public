# Deployment Guide

This document covers deployment scenarios, cloud storage configuration, and vector database setup.

---

## Table of Contents

1. [Deployment Scenarios](#deployment-scenarios)
2. [Personal Cloud Backup](#personal-cloud-backup)
3. [R2 Cloud Storage](#r2-cloud-storage)
4. [Vector Database Configuration](#vector-database-configuration)
5. [Environment Variables](#environment-variables)

---

## Deployment Scenarios

Disaster Clippy supports multiple deployment scenarios to fit different use cases and infrastructure setups.

### Scenario 1: Self-Hosted Web Server with Personal Cloud

**Use Case:** You have your own web server (VPS, dedicated server, etc.) and want to host Disaster Clippy with your own cloud storage for backups.

**Setup:**

```bash
# On your web server
git clone https://github.com/yourusername/disaster-clippy-public.git
cd disaster-clippy-public
pip install -r requirements.txt

# Configure .env
VECTOR_DB_MODE=local          # Use ChromaDB (no Pinecone needed)
BACKUP_PATH=/var/www/backups  # Your server storage location
OPENAI_API_KEY=your_key       # Or use EMBEDDING_MODE=local with Ollama

# Run the app
python app.py
```

**What you get:**
- Full admin UI at yourdomain.com/useradmin/
- Chat interface at yourdomain.com
- Sources stored locally in `/var/www/backups`
- Personal cloud sync for backup/restore
- Can download source packs from official R2 or create your own

---

### Scenario 2: Raspberry Pi Offline Node with Optional Cloud Backup

**Use Case:** Set up a fully offline disaster preparedness node on a Raspberry Pi 5, with optional cloud backup when internet is available.

**Hardware Requirements:**
- Raspberry Pi 5 (8GB RAM recommended)
- External SSD strongly recommended (not HDD)
- Battery backup for power outages (optional)

**Setup:**

```bash
# On Raspberry Pi
git clone https://github.com/yourusername/disaster-clippy-public.git
cd disaster-clippy-public
pip install -r requirements.txt

# Configure .env
VECTOR_DB_MODE=local                    # ChromaDB for offline vectors
BACKUP_PATH=/mnt/external/disaster_data # External hard drive
EMBEDDING_MODE=local                    # Use Ollama (no API needed)

# Install Ollama for offline AI
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b

# Run the app
python app.py
```

**Configure for Hybrid Online/Offline Operation:**

1. Go to http://192.168.1.x:8000/useradmin/settings
2. Set Connection Mode to **"Hybrid"**
3. Enable Personal Cloud Backup (optional, for syncing when online)

**Access:**
- Users on your local network connect via WiFi/LAN: `http://192.168.1.x:8000`
- Cloud backup syncs when internet IS available
- Falls back to local-only when offline

**Performance Notes:**
- Chat responses: 2-10 seconds with llama3.2:3b
- Can serve 5-10 concurrent users comfortably
- Indexing large ZIM files will be slow (hours vs minutes on a server)

---

### Scenario 3: Global Admin with Personal Testing Environment

**Use Case:** You're the global admin maintaining the official Disaster Clippy deployment, but also want a personal testing environment with your own cloud storage.

**Two-Tier Credential System:**

```bash
# .env (protected by .gitignore) - Official global admin credentials
R2_ACCESS_KEY_ID=your_global_admin_key
R2_SECRET_ACCESS_KEY=your_global_secret
R2_ENDPOINT_URL=https://your_account.r2.cloudflarestorage.com
R2_BUCKET_NAME=disaster-clippy-backups
VECTOR_DB_MODE=global  # Full write access to official buckets
```

**Personal Testing Configuration via Settings UI:**
- `local_settings.json` (also protected by .gitignore)
- Enable Personal Cloud Backup
- Use different bucket: `disaster-clippy-bryan-testing`
- Separate credentials from official bucket
- Use for development/testing without affecting production

**Priority Order:**
1. When personal_cloud is enabled in `local_settings.json`, it overrides `.env`
2. When personal_cloud is disabled, falls back to `.env` (global admin mode)
3. If neither configured, no cloud storage

---

### Scenario 4: Air-Gapped Emergency Deployment

**Use Case:** Complete offline deployment for emergency preparedness, no internet at all.

**Pre-Deployment Preparation (while online):**

1. Download essential ZIM files:
   - Wikipedia medicine (~500MB)
   - Wikihow (~2GB)
   - Appropedia (~200MB)

2. Download Ollama model:
   ```bash
   ollama pull llama3.2:3b
   ```

3. Package everything on USB drive or external SSD:
   ```
   USB Drive/
   |-- disaster-clippy/ (full git clone)
   |-- backups/ (ZIM files, HTML backups)
   |-- ollama/ (portable Ollama installation)
   ```

**Deployment (offline):**

```bash
# Copy from USB to target machine
cp -r /mnt/usb/disaster-clippy /opt/
cd /opt/disaster-clippy

# Configure for offline-only
VECTOR_DB_MODE=local
BACKUP_PATH=/opt/disaster-clippy/backups
EMBEDDING_MODE=local

# Run
python app.py
```

**Connection Mode:** Set to "Offline Only" in settings UI

**Storage Requirements:**
- Minimum viable: ~10GB (core sources + small AI model)
- Comprehensive: ~50-100GB (full Wikipedia + larger AI models)

---

## Personal Cloud Backup

### Security Architecture

**File Storage:**
- Credentials stored in `local_settings.json` (gitignored)
- Never committed to repository
- Masked in settings UI (only last 4 characters of access key shown)

**File Permissions (Linux/Mac):**
```bash
chmod 600 local_settings.json
chmod 600 .env
```

**Test Connection Feature:**
- Tests credentials WITHOUT saving
- Verifies bucket access
- Shows connection status before committing

### Supported Cloud Providers

| Provider | Endpoint Template | Typical Cost |
|----------|------------------|--------------|
| **Cloudflare R2** | `https://ACCOUNT-ID.r2.cloudflarestorage.com` | $0.015/GB storage, no egress fees |
| **AWS S3** | `https://s3.amazonaws.com` | $0.023/GB storage + egress |
| **Backblaze B2** | `https://s3.us-west-002.backblazeb2.com` | $0.005/GB storage, 1GB/day free egress |
| **DigitalOcean Spaces** | `https://nyc3.digitaloceanspaces.com` | $5/month for 250GB + 1TB transfer |
| **Custom S3-Compatible** | User-defined | Varies |

### Configuration via Settings UI

1. Navigate to: `/useradmin/settings`
2. Enable "Personal Cloud Backup"
3. Select provider (auto-fills endpoint)
4. Enter credentials
5. Click "Test Connection"
6. Save settings

### Configuration via local_settings.json

```json
{
  "personal_cloud": {
    "enabled": true,
    "provider": "r2",
    "endpoint_url": "https://YOUR-ACCOUNT-ID.r2.cloudflarestorage.com",
    "access_key_id": "your_access_key",
    "secret_access_key": "your_secret_key",
    "bucket_name": "my-disaster-backups",
    "region": "auto"
  }
}
```

### Using Your Own Web Server as Cloud Storage

If you have a web server with an S3-compatible API (like MinIO):

```json
{
  "personal_cloud": {
    "enabled": true,
    "provider": "custom",
    "endpoint_url": "https://yourdomain.com:9000",
    "access_key_id": "your_minio_key",
    "secret_access_key": "your_minio_secret",
    "bucket_name": "disaster-backups",
    "region": "us-east-1"
  }
}
```

---

## R2 Cloud Storage

### Two-Bucket Architecture

The R2 storage system uses two separate buckets for security:

| Bucket | Purpose | Railway Access | Global Admin |
|--------|---------|----------------|--------------|
| `disaster-clippy-backups` | Official content | Read only | Read/Write |
| `disaster-clippy-submissions` | User submissions | Write only | Read/Delete |

### R2 Storage Functions

**File:** `offline_tools/cloud/r2.py`

**Bucket Getters:**
- `get_backups_storage()` - Uses `R2_BACKUPS_BUCKET` (reads official content)
- `get_submissions_storage()` - Uses `R2_SUBMISSIONS_BUCKET` (writes user submissions)
- `get_r2_storage()` - Legacy single-bucket mode (backward compatible)

**Server-Side Copy Methods:**
- `copy_to_bucket(source_key, dest_bucket, dest_key)` - Cross-bucket copy
- `move_to_bucket(source_key, dest_bucket, dest_key)` - Copy + delete

**Helper Functions:**
- `approve_submission(submission_key, dest_source_id, dest_filename)` - Server-side approve
- `reject_submission(submission_key, reason)` - Move to rejected folder

### R2 Environment Variables

**Railway Deployment** (limited access tokens):

```bash
R2_ACCESS_KEY_ID=<your-key-id>
R2_SECRET_ACCESS_KEY=<your-secret>
R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
R2_BACKUPS_BUCKET=disaster-clippy-backups
R2_SUBMISSIONS_BUCKET=disaster-clippy-submissions
```

**Global Admin** (full access token):

```bash
R2_ACCESS_KEY_ID=<admin-key-id>
R2_SECRET_ACCESS_KEY=<admin-secret>
R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
R2_BACKUPS_BUCKET=disaster-clippy-backups
R2_SUBMISSIONS_BUCKET=disaster-clippy-submissions
```

### Cloudflare R2 Setup

1. Create bucket: `disaster-clippy-backups`
2. Create bucket: `disaster-clippy-submissions`
3. Create API token for Railway: Read on backups, Write on submissions
4. Create API token for Global Admin: Full access on both

The code is backward compatible - if you only set `R2_BUCKET_NAME`, it uses single-bucket mode.

---

## Vector Database Configuration

`VECTOR_DB_MODE` controls where vectors are stored:

| Mode | Storage | Use Case |
|------|---------|----------|
| `local` (default) | ChromaDB in `BACKUP_PATH/chroma/` | Local admin, offline use |
| `pinecone` | Pinecone cloud service | Railway deployment, global admin |

### Pinecone Variables

Only needed if `VECTOR_DB_MODE=pinecone`:

| Variable | Description |
|----------|-------------|
| `PINECONE_API_KEY` | Your Pinecone API key from console.pinecone.io |
| `PINECONE_ENVIRONMENT` | Region (e.g., us-east-1, gcp-starter) |
| `PINECONE_INDEX_NAME` | Index name (default: disaster-clippy) |

### Pinecone Sync API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/pinecone-status` | GET | Get Pinecone connection status and stats |
| `/pinecone-check-source/{source_id}` | GET | Check if source exists in Pinecone (returns vector count) |
| `/pinecone-source/{source_id}` | DELETE | Delete all vectors for a source (global admin only) |
| `/pinecone-compare` | POST | Compare local ChromaDB with Pinecone |
| `/pinecone-sync` | POST | Sync local vectors to Pinecone |

### Deployment Requirements

| Deployment | VECTOR_DB_MODE | Pinecone Keys |
|------------|----------------|---------------|
| Local admin | `local` | Not needed |
| Railway public | `pinecone` | Yes - in Railway env vars |
| Global admin | `pinecone` | Yes - for syncing to cloud |

---

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `OPENAI_API_KEY` | Embeddings and chat | Yes (unless EMBEDDING_MODE=local) |
| `ANTHROPIC_API_KEY` | Claude chat (optional) | No |
| `PINECONE_API_KEY` | Cloud vector DB | Global admin only |
| `R2_ACCESS_KEY_ID` | Cloud storage | For R2 upload/download |
| `R2_SECRET_ACCESS_KEY` | Cloud storage | For R2 upload/download |
| `ADMIN_MODE` | "local" or "global" | No (defaults to local) |
| `EMBEDDING_MODE` | "openai" or "local" | No (defaults to openai) |
| `VECTOR_DB_MODE` | "local" or "pinecone" | No (defaults to local) |
| `BACKUP_PATH` | Path to backup folder | No (defaults to ./backups) |
| `RAILWAY_PROXY_URL` | Railway proxy for cloud access | For local admins without keys |

---

## Related Documentation

- [Architecture](architecture.md) - System design and security modes
- [Admin Guide](admin-guide.md) - Setting up and running the admin panel

---

*Last Updated: December 2025*
