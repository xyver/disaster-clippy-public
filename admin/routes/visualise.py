"""
Knowledge Map Visualization API

Endpoints for generating and viewing 3D visualization of document embeddings.
Uses PCA to reduce 1536-dim vectors to 3D for Plotly scatter plot.
"""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import json
import gzip
import numpy as np

router = APIRouter(prefix="/api/visualise", tags=["Visualization"])


def get_local_config():
    """Get local config - imported here to avoid circular imports"""
    from admin.local_config import get_local_config as _get_config
    return _get_config()


def get_visualisation_folder() -> Optional[Path]:
    """Get the folder for visualization files."""
    config = get_local_config()
    backup_folder = config.get_backup_folder()
    if not backup_folder:
        return None
    vis_folder = Path(backup_folder) / "visualisation"
    vis_folder.mkdir(parents=True, exist_ok=True)

    # Migrate old files from backup root to visualisation folder
    backup_path = Path(backup_folder)
    old_main = backup_path / "_visualisation.json"
    if old_main.exists():
        new_main = vis_folder / "_visualisation.json"
        if not new_main.exists():
            old_main.rename(new_main)
            print(f"[Visualisation] Migrated _visualisation.json to visualisation folder")

    # Migrate per-source files
    for old_file in backup_path.glob("_visualisation_urls_*.json"):
        new_file = vis_folder / old_file.name
        if not new_file.exists():
            old_file.rename(new_file)
            print(f"[Visualisation] Migrated {old_file.name} to visualisation folder")

    for old_file in backup_path.glob("_visualisation_edges_*.json"):
        new_file = vis_folder / old_file.name
        if not new_file.exists():
            old_file.rename(new_file)
            print(f"[Visualisation] Migrated {old_file.name} to visualisation folder")

    return vis_folder


def get_visualisation_path() -> Optional[Path]:
    """Get the path for the main visualization JSON file."""
    vis_folder = get_visualisation_folder()
    if not vis_folder:
        return None
    return vis_folder / "_visualisation.json"


def get_job_manager():
    """Get job manager - imported here to avoid circular imports"""
    from admin.job_manager import get_job_manager as _get_manager
    return _get_manager()


# =============================================================================
# LINK EDGE BUILDING
# =============================================================================

def build_link_edges(points: List[Dict], sources: set, progress_callback=None) -> List[Dict]:
    """
    Build edges from internal links stored in _metadata.json files.

    Returns list of edges: [{"from": doc_id, "to": doc_id}, ...]
    """
    config = get_local_config()
    backup_folder = config.get_backup_folder()
    if not backup_folder:
        return []

    backup_path = Path(backup_folder)

    # Build URL-to-point-index mapping for fast lookup
    # Note: uses _url and _local_url fields (temporary, stripped before save)
    url_to_idx = {}
    for idx, point in enumerate(points):
        url = point.get("_url", "") or point.get("url", "")
        if url:
            # Store both with and without leading slash for matching
            url_to_idx[url] = idx
            if url.startswith("/"):
                url_to_idx[url[1:]] = idx
            else:
                url_to_idx["/" + url] = idx

    edges = []
    edge_set = set()  # Deduplicate edges

    # Also build mapping with local_url for ZIM sources
    local_url_to_idx = {}
    for idx, point in enumerate(points):
        local_url = point.get("_local_url", "") or point.get("local_url", "")
        if local_url:
            local_url_to_idx[local_url] = idx
            if local_url.startswith("/"):
                local_url_to_idx[local_url[1:]] = idx
            else:
                local_url_to_idx["/" + local_url] = idx

    # Load metadata from each source and build edges
    for source_id in sources:
        source_path = backup_path / source_id
        metadata_file = source_path / "_metadata.json"

        if not metadata_file.exists():
            continue

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            docs = metadata.get("documents", {})

            # Load manifest to get base_url for constructing full URLs from relative paths
            manifest_file = source_path / "_manifest.json"
            base_url = None
            if manifest_file.exists():
                try:
                    with open(manifest_file, 'r', encoding='utf-8') as mf:
                        manifest_data = json.load(mf)
                        base_url = manifest_data.get("base_url", "").rstrip("/")
                except:
                    pass
            
            print(f"[{source_id}] Loaded base_url: '{base_url}'")

            # Debug: Show sample URLs from ChromaDB for this source
            print(f"\n[{source_id}] Sample URLs in ChromaDB:")
            source_points = [p for p in points if p.get("source") == source_id]
            for p in source_points[:3]:
                print(f"  - url: '{p.get('url', '')}', local_url: '{p.get('local_url', '')}'")
            print(f"[{source_id}] Sample URLs in metadata:")
            docs_with_links = 0
            matched_links = 0
            failed_links = 0

            # Debug: sample first doc with links
            sample_logged = False
            sample_metadata_logged = 0
            docs_without_from_idx = 0

            for doc_id, doc_data in docs.items():
                internal_links = doc_data.get("internal_links", [])
                if not internal_links:
                    continue

                docs_with_links += 1

                # Log first 3 docs from metadata for comparison
                if sample_metadata_logged < 3:
                    print(f"  - url: '{doc_data.get('url', '')}', local_url: '{doc_data.get('local_url', '')}'")
                    sample_metadata_logged += 1

                # Find the source point index - try both url and local_url
                from_idx = None
                doc_url = doc_data.get("url", "")
                doc_local_url = doc_data.get("local_url", "")

                # If metadata's url field starts with /zim/, it's actually a local_url
                # Look it up in local_url_to_idx instead
                if doc_url.startswith("/zim/"):
                    from_idx = local_url_to_idx.get(doc_url)
                    if from_idx is None:
                        # Try with/without leading slash
                        if doc_url.startswith("/"):
                            from_idx = local_url_to_idx.get(doc_url[1:])
                        else:
                            from_idx = local_url_to_idx.get("/" + doc_url)
                # Otherwise try normal URL lookup
                elif doc_url in url_to_idx:
                    from_idx = url_to_idx[doc_url]
                elif doc_local_url in local_url_to_idx:
                    from_idx = local_url_to_idx[doc_local_url]

                if from_idx is None:
                    docs_without_from_idx += 1
                    # Log first failure to debug
                    if not sample_logged:
                        print(f"  [DEBUG] Can't find doc: url='{doc_url}', local_url='{doc_local_url}'")
                        sample_logged = True
                    continue

                # Resolve each link to a target point
                for link_url in internal_links:
                    to_idx = None

                    # If link is a relative path and we have base_url, construct full URL
                    if base_url and link_url.startswith("/") and not link_url.startswith("/zim/") and not link_url.startswith("/backup/"):
                        # Relative path like /Bitcoin -> https://en.bitcoin.it/wiki/Bitcoin
                        full_url = base_url + link_url
                        to_idx = url_to_idx.get(full_url)


                    # If link starts with /zim/, look it up as local_url
                    if to_idx is None and link_url.startswith("/zim/"):
                        to_idx = local_url_to_idx.get(link_url)
                        if to_idx is None:
                            if link_url.startswith("/"):
                                to_idx = local_url_to_idx.get(link_url[1:])
                            else:
                                to_idx = local_url_to_idx.get("/" + link_url)
                    elif to_idx is None:
                        # Otherwise try normal URL lookup
                        to_idx = url_to_idx.get(link_url)
                        if to_idx is None:
                            # Try with/without leading slash
                            if link_url.startswith("/"):
                                to_idx = url_to_idx.get(link_url[1:])
                            else:
                                to_idx = url_to_idx.get("/" + link_url)

                        # Also try local_url mapping as fallback
                        if to_idx is None:
                            to_idx = local_url_to_idx.get(link_url)
                            if to_idx is None:
                                if link_url.startswith("/"):
                                    to_idx = local_url_to_idx.get(link_url[1:])
                                else:
                                    to_idx = local_url_to_idx.get("/" + link_url)

                    if to_idx is not None and to_idx != from_idx:
                        # Create edge key to avoid duplicates
                        edge_key = (min(from_idx, to_idx), max(from_idx, to_idx))
                        if edge_key not in edge_set:
                            edge_set.add(edge_key)
                            edges.append({
                                "from": from_idx,
                                "to": to_idx
                            })
                            matched_links += 1
                    else:
                        failed_links += 1

            print(f"[{source_id}] {docs_with_links} docs with links, {matched_links} matched, {failed_links} failed, {docs_without_from_idx} docs not found in points")

        except Exception as e:
            print(f"Error loading metadata for {source_id}: {e}")
            continue

    return edges


# =============================================================================
# VISUALIZATION GENERATION
# =============================================================================

def generate_visualisation_job(
    progress_callback=None,
    cancel_checker=None
) -> Dict[str, Any]:
    """
    Background job to generate visualization data.

    Pulls all vectors from ChromaDB, runs PCA to reduce to 3D,
    and saves to _visualisation.json.

    Args:
        progress_callback: Function(current, total, message) for progress
        cancel_checker: Function() returns True if job should cancel

    Returns:
        Dict with success status and stats
    """
    from sklearn.decomposition import PCA
    from offline_tools.vectordb import get_vector_store

    output_path = get_visualisation_path()
    if not output_path:
        return {"success": False, "error": "No backup folder configured"}

    try:
        if progress_callback:
            progress_callback(0, 100, "Connecting to ChromaDB...")

        # Get vector store in read_only mode to skip embedding model loading
        store = get_vector_store(mode="local", read_only=True)
        total_docs = store.collection.count()

        if total_docs == 0:
            return {"success": False, "error": "No documents in ChromaDB"}

        if progress_callback:
            progress_callback(5, 100, f"Found {total_docs:,} documents, fetching embeddings...")

        # Fetch embeddings in batches to show progress (ChromaDB get() is blocking)
        BATCH_SIZE = 10000
        all_ids = []
        all_embeddings = []
        all_metadatas = []

        for offset in range(0, total_docs, BATCH_SIZE):
            # Check for cancellation
            if cancel_checker and cancel_checker():
                return {"status": "cancelled"}

            batch_num = (offset // BATCH_SIZE) + 1
            total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
            pct = 5 + int((offset / total_docs) * 15)  # 5% to 20%

            if progress_callback:
                progress_callback(pct, 100, f"Fetching batch {batch_num}/{total_batches} ({offset:,}/{total_docs:,})...")

            result = store.collection.get(
                include=["embeddings", "metadatas"],
                limit=BATCH_SIZE,
                offset=offset
            )

            all_ids.extend(result.get("ids", []))
            all_embeddings.extend(result.get("embeddings", []))
            all_metadatas.extend(result.get("metadatas", []))

        ids = all_ids
        embeddings = all_embeddings
        metadatas = all_metadatas

        if not embeddings:
            return {"success": False, "error": "No embeddings found in ChromaDB"}

        if progress_callback:
            progress_callback(20, 100, f"Loaded {len(embeddings):,} embeddings, running PCA...")

        # Check for cancellation
        if cancel_checker and cancel_checker():
            return {"status": "cancelled"}

        # Convert to numpy array for PCA
        embedding_matrix = np.array(embeddings)

        # Run PCA to reduce to 3D
        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(embedding_matrix)

        # Calculate variance explained
        variance_explained = pca.explained_variance_ratio_.tolist()
        total_variance = sum(variance_explained)

        if progress_callback:
            progress_callback(60, 100, f"PCA complete (variance: {total_variance:.1%}), building output...")

        # Check for cancellation
        if cancel_checker and cancel_checker():
            return {"status": "cancelled"}

        # Build points list
        # OPTIMIZATION: Split into per-source files for lazy loading
        # - Core: x,y,z (3 decimals), source, truncated title - always loaded
        # - URLs: per-source files, loaded on-demand when clicking points from that source
        # - Edges: per-source files, loaded when enabling links for that source
        points = []
        urls_by_source = {}  # {source_id: {point_index: url}}
        sources_set = set()

        for i, doc_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            source = metadata.get("source", "unknown")
            sources_set.add(source)

            # Get URL - prefer online URL, fall back to local_url
            url = metadata.get("url", "") or metadata.get("local_url", "")

            # Truncate title to save space (max 80 chars)
            title = metadata.get("title", "Untitled")
            if len(title) > 80:
                title = title[:77] + "..."

            # Compact point format - coordinates rounded to 3 decimal places
            # Only essential fields for rendering
            point = {
                "x": round(float(coords_3d[i][0]), 3),
                "y": round(float(coords_3d[i][1]), 3),
                "z": round(float(coords_3d[i][2]), 3),
                "id": doc_id,
                "title": title,
                "source": source,
                # Keep url and local_url for edge building (will be stripped before save)
                "_url": url,
                "_local_url": metadata.get("local_url", "")
            }
            points.append(point)

            # Store URL in per-source lookup (only if exists)
            if url:
                if source not in urls_by_source:
                    urls_by_source[source] = {}
                urls_by_source[source][i] = url

            # Progress update every 1000 docs
            if i > 0 and i % 1000 == 0:
                if progress_callback:
                    pct = 60 + int((i / len(ids)) * 30)
                    progress_callback(pct, 100, f"Processing {i}/{len(ids)} documents...")

                if cancel_checker and cancel_checker():
                    return {"status": "cancelled"}

        if progress_callback:
            progress_callback(85, 100, "Building link edges...")

        # Build source counts
        source_counts = {}
        for point in points:
            src = point["source"]
            source_counts[src] = source_counts.get(src, 0) + 1

        # Build edges from internal links
        edges = build_link_edges(points, sources_set, progress_callback)

        if progress_callback:
            progress_callback(95, 100, "Saving visualization data...")

        # Group edges by source (based on 'from' point's source)
        edges_by_source = {}
        for edge in edges:
            from_idx = edge["from"]
            if from_idx < len(points):
                source = points[from_idx]["source"]
                if source not in edges_by_source:
                    edges_by_source[source] = []
                edges_by_source[source].append(edge)

        # Strip temporary URL fields from points (used only for edge building)
        for point in points:
            point.pop("_url", None)
            point.pop("_local_url", None)

        # PER-SOURCE FILE STRUCTURE:
        # - Core: _visualisation.json (always loaded)
        # - URLs: _visualisation_urls_{source_id}.json (per source, loaded on click)
        # - Edges: _visualisation_edges_{source_id}.json (per source, loaded on toggle)

        generated_at = datetime.now().isoformat()

        # File 1: Core visualization data (points without URLs/edges)
        core_output = {
            "generated_at": generated_at,
            "algorithm": "pca",
            "point_count": len(points),
            "edge_count": len(edges),
            "sources": sorted(list(sources_set)),
            "source_counts": source_counts,
            "variance_explained": variance_explained,
            "total_variance": total_variance,
            "points": points
            # Note: URLs and edges loaded separately per-source
        }

        # Save core file (atomic write)
        temp_path = output_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(core_output, f, separators=(',', ':'))  # Compact JSON
        temp_path.replace(output_path)
        core_size_mb = output_path.stat().st_size / (1024 * 1024)

        # Save per-source URL files
        urls_total_size = 0
        for source_id, source_urls in urls_by_source.items():
            urls_output = {
                "generated_at": generated_at,
                "source": source_id,
                "urls": source_urls  # {point_index: url}
            }
            # Sanitize source_id for filename (replace problematic chars)
            safe_source = source_id.replace("/", "_").replace("\\", "_")
            urls_path = output_path.parent / f"_visualisation_urls_{safe_source}.json"
            temp_path = urls_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(urls_output, f, separators=(',', ':'))
            temp_path.replace(urls_path)
            urls_total_size += urls_path.stat().st_size

        # Save per-source edge files
        edges_total_size = 0
        for source_id, source_edges in edges_by_source.items():
            edges_output = {
                "generated_at": generated_at,
                "source": source_id,
                "edge_count": len(source_edges),
                "edges": source_edges
            }
            safe_source = source_id.replace("/", "_").replace("\\", "_")
            edges_path = output_path.parent / f"_visualisation_edges_{safe_source}.json"
            temp_path = edges_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(edges_output, f, separators=(',', ':'))
            temp_path.replace(edges_path)
            edges_total_size += edges_path.stat().st_size

        # Calculate total file sizes
        urls_size_mb = urls_total_size / (1024 * 1024)
        edges_size_mb = edges_total_size / (1024 * 1024)
        total_size_mb = core_size_mb + urls_size_mb + edges_size_mb

        print(f"[Visualisation] Saved: core={core_size_mb:.1f}MB, urls={urls_size_mb:.1f}MB ({len(urls_by_source)} files), edges={edges_size_mb:.1f}MB ({len(edges_by_source)} files)")

        if progress_callback:
            progress_callback(100, 100, "Visualization complete!")

        return {
            "success": True,
            "point_count": len(points),
            "source_count": len(sources_set),
            "file_size_mb": round(total_size_mb, 2),
            "variance_explained": total_variance
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def generate_and_publish_visualisation_job(
    progress_callback=None,
    cancel_checker=None
) -> Dict[str, Any]:
    """
    Chained job: Generate visualization then publish to R2.
    Used when triggered from Pinecone push.

    Args:
        progress_callback: Function(current, total, message) for progress
        cancel_checker: Function() returns True if job should cancel

    Returns:
        Dict with success status and stats
    """
    # Step 1: Generate visualization (0-90% of progress)
    def gen_progress(current, total, message):
        if progress_callback:
            # Map to 0-90% range
            scaled_current = int((current / total) * 90)
            progress_callback(scaled_current, 100, f"Generating: {message}")

    result = generate_visualisation_job(
        progress_callback=gen_progress,
        cancel_checker=cancel_checker
    )

    if not result.get("success"):
        return result

    # Check for cancellation
    if cancel_checker and cancel_checker():
        return {"status": "cancelled"}

    # Step 2: Publish to R2 (90-100% of progress)
    if progress_callback:
        progress_callback(90, 100, "Publishing to R2...")

    try:
        from offline_tools.cloud.r2 import get_r2_storage
        storage = get_r2_storage()

        if not storage.is_configured():
            print("[Visualisation] R2 not configured, skipping publish")
            result["publish"] = {"published": False, "reason": "R2 not configured"}
            return result

        conn_status = storage.test_connection()
        if not conn_status["connected"]:
            print(f"[Visualisation] R2 not connected: {conn_status.get('error')}")
            result["publish"] = {"published": False, "reason": "R2 connection failed"}
            return result

        vis_path = get_visualisation_path()
        if not vis_path or not vis_path.exists():
            result["publish"] = {"published": False, "reason": "Visualization file not found"}
            return result

        # Build list of files: core + all per-source files
        backup_folder = vis_path.parent
        files_to_upload = [(vis_path, "published/visualisation.json")]

        # Find all per-source URL files
        for urls_file in backup_folder.glob("_visualisation_urls_*.json"):
            r2_key = f"published/{urls_file.name}"
            files_to_upload.append((urls_file, r2_key))

        # Find all per-source edge files
        for edges_file in backup_folder.glob("_visualisation_edges_*.json"):
            r2_key = f"published/{edges_file.name}"
            files_to_upload.append((edges_file, r2_key))

        uploaded_files = []
        for local_path, r2_key in files_to_upload:
            if local_path.exists():
                success = storage.upload_file(str(local_path), r2_key)
                if success:
                    uploaded_files.append(r2_key)
                    print(f"[Visualisation] Published {r2_key} to R2")
                else:
                    print(f"[Visualisation] Failed to upload {r2_key}")

        if uploaded_files:
            file_size_mb = result.get("file_size_mb", 0)
            point_count = result.get("point_count", 0)
            print(f"[Visualisation] Published {len(uploaded_files)} files to R2 ({file_size_mb:.2f} MB, {point_count} points)")
            result["publish"] = {"published": True, "files": uploaded_files}
        else:
            print("[Visualisation] Failed to upload any files to R2")
            result["publish"] = {"published": False, "reason": "Upload failed"}

        if progress_callback:
            progress_callback(100, 100, "Complete! Visualization published to R2")

        return result

    except Exception as e:
        print(f"[Visualisation] Error publishing: {e}")
        result["publish"] = {"published": False, "error": str(e)}
        return result


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.get("/status")
async def get_visualisation_status():
    """
    Get current visualization status.

    Returns info about existing visualization file and any active generation job.
    """
    result = {
        "has_data": False,
        "generated_at": None,
        "point_count": 0,
        "edge_count": 0,
        "sources": [],
        "source_counts": {},
        "file_size_mb": 0,
        "job_active": False,
        "job_progress": 0,
        "job_message": ""
    }

    # Check for existing visualization file
    vis_path = get_visualisation_path()
    if vis_path and vis_path.exists():
        try:
            with open(vis_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            result["has_data"] = True
            result["generated_at"] = data.get("generated_at")
            result["point_count"] = data.get("point_count", 0)
            result["edge_count"] = data.get("edge_count", 0)
            result["sources"] = data.get("sources", [])
            result["source_counts"] = data.get("source_counts", {})
            result["algorithm"] = data.get("algorithm", "pca")
            result["variance_explained"] = data.get("total_variance", 0)
            result["file_size_mb"] = round(vis_path.stat().st_size / (1024 * 1024), 2)
        except Exception as e:
            print(f"Error reading visualization file: {e}")

    # Check for active job
    try:
        manager = get_job_manager()
        active_jobs = manager.get_active_jobs()
        for job in active_jobs:
            if job.get("job_type") == "visualisation":
                result["job_active"] = True
                result["job_progress"] = job.get("progress", 0)
                result["job_message"] = job.get("message", "")
                result["job_id"] = job.get("id")
                break
    except Exception as e:
        print(f"Error checking job status: {e}")

    return result


@router.get("/data")
async def get_visualisation_data(
    sources: Optional[str] = None
):
    """
    Get visualization data for rendering.

    Args:
        sources: Optional comma-separated list of sources to filter

    Returns:
        Full visualization data or filtered subset
    """
    vis_path = get_visualisation_path()
    if not vis_path or not vis_path.exists():
        raise HTTPException(404, "No visualization data found. Generate it first.")

    try:
        with open(vis_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Filter by sources if specified
        if sources:
            source_list = [s.strip() for s in sources.split(",")]
            data["points"] = [
                p for p in data["points"]
                if p.get("source") in source_list
            ]
            data["point_count"] = len(data["points"])

        return data

    except Exception as e:
        raise HTTPException(500, f"Error loading visualization data: {e}")


@router.get("/urls/{source_id}")
async def get_visualisation_urls(source_id: str):
    """
    Get URL lookup data for a specific source.

    Loaded on-demand when user clicks a point from this source.
    Returns: {source: source_id, urls: {point_index: url, ...}}
    """
    vis_folder = get_visualisation_folder()
    if not vis_folder:
        raise HTTPException(404, "No backup folder configured")

    # Sanitize source_id for filename
    safe_source = source_id.replace("/", "_").replace("\\", "_")
    urls_path = vis_folder / f"_visualisation_urls_{safe_source}.json"

    if not urls_path.exists():
        # Return empty data if source has no URLs (not an error)
        return {"source": source_id, "urls": {}}

    try:
        with open(urls_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(500, f"Error loading URL data: {e}")


@router.get("/edges/{source_id}")
async def get_visualisation_edges(source_id: str):
    """
    Get edge/connection data for a specific source.

    Loaded on-demand when user enables connection lines for this source.
    Returns: {source: source_id, edges: [{from: idx, to: idx}, ...], edge_count: N}
    """
    vis_folder = get_visualisation_folder()
    if not vis_folder:
        raise HTTPException(404, "No backup folder configured")

    # Sanitize source_id for filename
    safe_source = source_id.replace("/", "_").replace("\\", "_")
    edges_path = vis_folder / f"_visualisation_edges_{safe_source}.json"

    if not edges_path.exists():
        # Return empty data if source has no edges (not an error)
        return {"source": source_id, "edges": [], "edge_count": 0}

    try:
        with open(edges_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(500, f"Error loading edge data: {e}")


@router.post("/generate")
async def generate_visualisation():
    """
    Start background job to generate visualization.

    Returns job ID for tracking progress.
    """
    # Check if backup folder is configured
    config = get_local_config()
    if not config.get_backup_folder():
        raise HTTPException(400, "No backup folder configured. Set it in Settings first.")

    # Check if a visualization job is already running
    manager = get_job_manager()
    active_jobs = manager.get_active_jobs()
    for job in active_jobs:
        if job.get("job_type") == "visualisation":
            return {
                "status": "already_running",
                "job_id": job.get("id"),
                "message": "Visualization generation already in progress"
            }

    try:
        # Submit background job
        job_id = manager.submit(
            job_type="visualisation",
            source_id="_system",  # System-level job, not source-specific
            func=generate_visualisation_job
        )

        return {
            "status": "started",
            "job_id": job_id,
            "message": "Visualization generation started"
        }

    except ValueError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error starting visualization job: {e}")


@router.delete("/data")
async def delete_visualisation():
    """Delete the visualization data file and all per-source files."""
    vis_folder = get_visualisation_folder()
    vis_path = get_visualisation_path()
    if not vis_path or not vis_path.exists():
        return {"status": "not_found", "message": "No visualization data to delete"}

    try:
        deleted_count = 0
        # Delete main file
        vis_path.unlink()
        deleted_count += 1

        # Delete all per-source URL and edge files
        if vis_folder:
            for f in vis_folder.glob("_visualisation_urls_*.json"):
                f.unlink()
                deleted_count += 1
            for f in vis_folder.glob("_visualisation_edges_*.json"):
                f.unlink()
                deleted_count += 1

        return {"status": "deleted", "message": f"Deleted {deleted_count} visualization files"}
    except Exception as e:
        raise HTTPException(500, f"Error deleting visualization: {e}")


@router.post("/publish")
async def publish_to_r2():
    """
    Publish existing visualization files to R2 for public access.

    Uploads core file plus all per-source URL and edge files.
    Does NOT regenerate - use "Regenerate Visualization" button first if needed.
    """
    from offline_tools.cloud.r2 import get_r2_storage
    import glob

    # Check if local visualization exists
    vis_path = get_visualisation_path()
    if not vis_path or not vis_path.exists():
        raise HTTPException(400, "No local visualization found. Generate one first using 'Regenerate Visualization'.")

    backup_folder = vis_path.parent

    # Build list of files to upload: core + all per-source files
    files_to_upload = [(vis_path, "published/visualisation.json")]

    # Find all per-source URL files
    for urls_file in backup_folder.glob("_visualisation_urls_*.json"):
        r2_key = f"published/{urls_file.name}"
        files_to_upload.append((urls_file, r2_key))

    # Find all per-source edge files
    for edges_file in backup_folder.glob("_visualisation_edges_*.json"):
        r2_key = f"published/{edges_file.name}"
        files_to_upload.append((edges_file, r2_key))

    try:
        # Load core file to get stats
        with open(vis_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        point_count = data.get("point_count", 0)
        edge_count = data.get("edge_count", 0)
        source_count = len(data.get("sources", []))

        # Calculate total file size
        total_size_mb = 0
        for local_path, _ in files_to_upload:
            if local_path.exists():
                total_size_mb += local_path.stat().st_size / (1024 * 1024)
        total_size_mb = round(total_size_mb, 2)

        # Get R2 storage
        storage = get_r2_storage()

        if not storage.is_configured():
            raise HTTPException(400, "R2 cloud storage not configured")

        # Test connection
        conn_status = storage.test_connection()
        if not conn_status["connected"]:
            raise HTTPException(500, f"R2 connection failed: {conn_status.get('error', 'Unknown error')}")

        # Upload all files to R2
        uploaded_files = []
        for local_path, r2_key in files_to_upload:
            if local_path.exists():
                success = storage.upload_file(str(local_path), r2_key)
                if not success:
                    raise Exception(f"Upload failed for {r2_key}")
                uploaded_files.append(r2_key)
                print(f"Published {r2_key} to R2")

        print(f"Published visualization to R2 ({total_size_mb} MB total, {len(uploaded_files)} files, {point_count} points, {edge_count} edges)")

        return {
            "status": "success",
            "message": f"Visualization published to R2 ({len(uploaded_files)} files for {source_count} sources)",
            "files": uploaded_files,
            "point_count": point_count,
            "edge_count": edge_count,
            "file_size_mb": total_size_mb
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Error publishing visualization: {e}")
