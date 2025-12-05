# Pipeline Testing Checklist

Manual testing checklist for the source ingestion pipeline. Run through these steps to verify the 5-step wizard and file creation tools work correctly.

---

## Prerequisites

- [ ] Admin panel accessible at `/useradmin/`
- [ ] Backup folder configured in Settings
- [ ] At least one test source available (small wiki or HTML folder)

---

## Test 1: Fresh Source Creation (HTML Backup)

Test creating a new source from scratch using the HTML backup scraper.

### Step 1: Create Backup
- [ ] Go to Source Tools -> select source or create new
- [ ] Click "Create Backup" or run HTML scraper
- [ ] Verify `backup_manifest.json` created in `BACKUP_PATH/{source_id}/`
- [ ] Verify HTML files saved in `pages/` subfolder
- [ ] Verify backup_manifest.json contains:
  - `schema_version: 3`
  - `source_id` matches folder name
  - `pages` dict with URL -> filename mappings

### Step 2: Create Metadata
- [ ] Click "Create Metadata" in Source Tools
- [ ] Verify `_metadata.json` created
- [ ] Verify contains:
  - `schema_version: 3`
  - `document_count` matches number of pages
  - `documents` dict with doc entries

### Step 3: Create Index
- [ ] Click "Create Index" in Source Tools
- [ ] Verify `_manifest.json` created/updated
- [ ] Verify `_index.json` created
- [ ] Verify `_vectors.json` created
- [ ] Check `_manifest.json` contains:
  - `schema_version: 3`
  - File size fields populated
  - `source_type` set correctly

### Step 4: Verify Status Boxes
- [ ] All 5 status boxes show green/complete:
  - Config (has _manifest.json)
  - Backup (has backup files)
  - Metadata (has _metadata.json)
  - Embeddings (has _vectors.json)
  - License (license field set, ideally verified)

### Step 5: Test Search
- [ ] Go to main chat interface
- [ ] Search for content from the indexed source
- [ ] Verify results appear with correct source attribution

---

## Test 2: ZIM File Source

Test indexing from an existing ZIM file.

- [ ] Place a .zim file in `BACKUP_PATH/{source_id}/`
- [ ] Go to Source Tools, source should be auto-discovered
- [ ] Run "Create Index"
- [ ] Verify all schema files created:
  - `_manifest.json`
  - `_metadata.json`
  - `_index.json`
  - `_vectors.json`
- [ ] Verify status boxes all green
- [ ] Test search returns ZIM content

---

## Test 3: PDF Collection

Test the PDF collection workflow.

- [ ] Create folder `BACKUP_PATH/{source_id}/`
- [ ] Add PDF files to the folder
- [ ] Create `_collection.json` with collection metadata
- [ ] Run "Create Index"
- [ ] Verify schema files created
- [ ] Verify PDFs are chunked and indexed
- [ ] Test search returns PDF content

---

## Test 4: Source Rename

Test that renaming preserves all files correctly.

- [ ] Select an indexed source
- [ ] Use "Rename Source" tool
- [ ] Verify:
  - Folder renamed
  - All schema files preserved
  - `source_id` updated inside files
  - ChromaDB entries updated
  - Old folder deleted

---

## Test 5: Cleanup Redundant Files

Test migration cleanup on a source with legacy files.

- [ ] If you have a source with old format files, select it
- [ ] Run "Cleanup Redundant Files"
- [ ] Verify legacy files deleted:
  - `{source_id}_metadata.json`
  - `{source_id}_backup_manifest.json`
  - `{source_id}_source.json`
  - etc.
- [ ] Verify schema files preserved:
  - `_manifest.json`
  - `_metadata.json`
  - `_index.json`
  - `_vectors.json`
  - `backup_manifest.json`

---

## Test 6: Cloud Download (if R2 configured)

Test downloading a source pack from the cloud.

- [ ] Go to Sources -> Cloud tab
- [ ] Select a cloud source
- [ ] Click "Download"
- [ ] Verify all files downloaded to correct locations
- [ ] Verify ChromaDB populated
- [ ] Test search returns cloud content

---

## Schema File Verification

For any created source, verify file contents match schema:

### _manifest.json
```json
{
  "schema_version": 3,
  "source_id": "...",
  "name": "...",
  "description": "...",
  "license": "...",
  "license_verified": false,
  "base_url": "...",
  "source_type": "html|zim|pdf",
  "total_docs": 0,
  "total_chars": 0,
  ...
}
```

### _metadata.json
```json
{
  "schema_version": 3,
  "source_id": "...",
  "document_count": 0,
  "total_chars": 0,
  "last_updated": "...",
  "documents": {
    "doc_id": {
      "title": "...",
      "url": "...",
      "content_hash": "...",
      "char_count": 0
    }
  }
}
```

### _index.json
```json
{
  "schema_version": 3,
  "source_id": "...",
  "document_count": 0,
  "created_at": "...",
  "documents": {
    "doc_id": {
      "title": "...",
      "url": "...",
      "content": "..."
    }
  }
}
```

### _vectors.json
```json
{
  "schema_version": 3,
  "source_id": "...",
  "embedding_model": "...",
  "dimensions": 1536,
  "document_count": 0,
  "created_at": "...",
  "vectors": {
    "doc_id": [0.1, 0.2, ...]
  }
}
```

### backup_manifest.json
```json
{
  "schema_version": 3,
  "source_id": "...",
  "base_url": "...",
  "scraper_type": "...",
  "created_at": "...",
  "last_updated": "...",
  "pages": {
    "url": "filename.html"
  },
  "total_pages": 0
}
```

---

## Common Issues

### Status box shows red but files exist
- Check file is in correct location (`BACKUP_PATH/{source_id}/`)
- Check filename matches schema (e.g., `_metadata.json` not `{source_id}_metadata.json`)
- Check `schema_version` field exists and equals 3

### Search returns no results
- Verify ChromaDB has vectors (check `_vectors.json` document_count)
- Check source_id in vector metadata matches
- Try rebuilding index

### "Source not found" errors
- Verify `_manifest.json` exists in source folder
- Check source_id matches folder name

---

*Created: December 2025*
