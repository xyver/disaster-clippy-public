"""
Knowledge Map Visualization API

Endpoints for generating and viewing 3D visualization of document embeddings.
Uses PCA to reduce 1536-dim vectors to 3D for Plotly scatter plot.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import json
import numpy as np

router = APIRouter(prefix="/api/visualise", tags=["Visualization"])


def get_local_config():
    """Get local config - imported here to avoid circular imports"""
    from admin.local_config import get_local_config as _get_config
    return _get_config()


def get_visualisation_path() -> Optional[Path]:
    """Get the path for the visualization JSON file."""
    config = get_local_config()
    backup_folder = config.get_backup_folder()
    if not backup_folder:
        return None
    return Path(backup_folder) / "_visualisation.json"


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
    url_to_idx = {}
    for idx, point in enumerate(points):
        url = point.get("url", "")
        if url:
            # Store both with and without leading slash for matching
            url_to_idx[url] = idx
            if url.startswith("/"):
                url_to_idx[url[1:]] = idx
            else:
                url_to_idx["/" + url] = idx

    edges = []
    edge_set = set()  # Deduplicate edges

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

            for doc_id, doc_data in docs.items():
                internal_links = doc_data.get("internal_links", [])
                if not internal_links:
                    continue

                # Find the source point index
                from_idx = None
                doc_url = doc_data.get("url", "")
                if doc_url in url_to_idx:
                    from_idx = url_to_idx[doc_url]

                if from_idx is None:
                    continue

                # Resolve each link to a target point
                for link_url in internal_links:
                    to_idx = url_to_idx.get(link_url)
                    if to_idx is None:
                        # Try with/without leading slash
                        if link_url.startswith("/"):
                            to_idx = url_to_idx.get(link_url[1:])
                        else:
                            to_idx = url_to_idx.get("/" + link_url)

                    if to_idx is not None and to_idx != from_idx:
                        # Create edge key to avoid duplicates
                        edge_key = (min(from_idx, to_idx), max(from_idx, to_idx))
                        if edge_key not in edge_set:
                            edge_set.add(edge_key)
                            edges.append({
                                "from": from_idx,
                                "to": to_idx
                            })

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

        # Get vector store
        store = get_vector_store(mode="local")
        stats = store.get_stats()
        total_docs = stats.get("total_documents", 0)

        if total_docs == 0:
            return {"success": False, "error": "No documents in ChromaDB"}

        if progress_callback:
            progress_callback(5, 100, f"Found {total_docs} documents, fetching embeddings...")

        # Check for cancellation
        if cancel_checker and cancel_checker():
            return {"status": "cancelled"}

        # Fetch all documents with embeddings
        result = store.collection.get(
            include=["embeddings", "metadatas"]
        )

        ids = result.get("ids", [])
        embeddings = result.get("embeddings", [])
        metadatas = result.get("metadatas", [])

        if embeddings is None or len(embeddings) == 0:
            return {"success": False, "error": "No embeddings found in ChromaDB"}

        if progress_callback:
            progress_callback(20, 100, f"Loaded {len(embeddings)} embeddings, running PCA...")

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
        points = []
        sources_set = set()

        for i, doc_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            source = metadata.get("source", "unknown")
            sources_set.add(source)

            # Parse categories if JSON string
            categories = metadata.get("categories", [])
            if isinstance(categories, str):
                try:
                    categories = json.loads(categories)
                except:
                    categories = []

            point = {
                "x": float(coords_3d[i][0]),
                "y": float(coords_3d[i][1]),
                "z": float(coords_3d[i][2]),
                "id": doc_id,
                "title": metadata.get("title", "Untitled"),
                "source": source,
                "doc_type": metadata.get("doc_type", "article"),
                "url": metadata.get("url", ""),
                "local_url": metadata.get("local_url", "")
            }
            points.append(point)

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

        # Build output structure
        output = {
            "generated_at": datetime.now().isoformat(),
            "algorithm": "pca",
            "point_count": len(points),
            "edge_count": len(edges),
            "sources": sorted(list(sources_set)),
            "source_counts": source_counts,
            "variance_explained": variance_explained,
            "total_variance": total_variance,
            "points": points,
            "edges": edges
        }

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f)

        # Calculate file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        if progress_callback:
            progress_callback(100, 100, "Visualization complete!")

        return {
            "success": True,
            "point_count": len(points),
            "source_count": len(sources_set),
            "file_size_mb": round(file_size_mb, 2),
            "variance_explained": total_variance
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


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
    """Delete the visualization data file."""
    vis_path = get_visualisation_path()
    if not vis_path or not vis_path.exists():
        return {"status": "not_found", "message": "No visualization data to delete"}

    try:
        vis_path.unlink()
        return {"status": "deleted", "message": "Visualization data deleted"}
    except Exception as e:
        raise HTTPException(500, f"Error deleting visualization: {e}")
