"""
Job Parameter Schemas for Job Builder

Defines chainable jobs and their parameters for the visual job chain builder.
Used by /admin/job-builder to generate the UI and validate job chains.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class JobParam:
    """Definition of a single job parameter"""
    name: str
    type: str  # "string", "int", "float", "bool", "select"
    label: str
    default: Any = None
    required: bool = False
    description: str = ""
    options: List[str] = field(default_factory=list)  # For "select" type
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class JobSchema:
    """Definition of a chainable job"""
    job_type: str
    label: str
    description: str
    category: str  # "content", "indexing", "maintenance", "cloud"
    params: List[JobParam] = field(default_factory=list)
    weight: int = 50  # Default progress weight in combined jobs
    requires_source: bool = True  # Most jobs need source_id
    endpoint: str = ""  # API endpoint to call

    # Validation rules
    should_run_before: List[str] = field(default_factory=list)  # This job should run before these
    should_run_after: List[str] = field(default_factory=list)   # This job should run after these
    conflicts_with: List[str] = field(default_factory=list)     # Cannot chain with these


# =============================================================================
# JOB SCHEMA DEFINITIONS
# =============================================================================

JOB_SCHEMAS: Dict[str, JobSchema] = {

    # -------------------------------------------------------------------------
    # CONTENT JOBS
    # -------------------------------------------------------------------------

    "scrape": JobSchema(
        job_type="scrape",
        label="Scrape",
        description="Scrape content from a website URL or sitemap",
        category="content",
        endpoint="/api/start-scrape",
        weight=30,
        params=[
            JobParam("url", "string", "URL", required=True,
                     description="Website URL or sitemap XML to scrape"),
            JobParam("page_limit", "int", "Page Limit", default=100,
                     min_value=1, max_value=10000,
                     description="Maximum pages to scrape"),
            JobParam("include_assets", "bool", "Include Assets", default=False,
                     description="Download images, CSS, JS files"),
            JobParam("follow_links", "bool", "Follow Links", default=True,
                     description="Follow internal links to find more pages"),
            JobParam("max_depth", "int", "Max Depth", default=3,
                     min_value=1, max_value=10,
                     description="Maximum link depth to follow"),
            JobParam("request_delay", "float", "Request Delay", default=0.5,
                     min_value=0.1, max_value=5.0,
                     description="Delay between requests (seconds)")
        ],
        should_run_before=["metadata", "index_online", "index_offline"]
    ),

    "metadata": JobSchema(
        job_type="metadata",
        label="Metadata",
        description="Generate document metadata (titles, descriptions, word counts)",
        category="content",
        endpoint="/api/generate-metadata",
        weight=20,
        params=[
            JobParam("language_filter", "string", "Language Filter", default="",
                     description="ISO code (en, es, fr) to filter ZIM articles"),
            JobParam("resume", "bool", "Resume", default=False,
                     description="Resume from checkpoint if available")
        ],
        should_run_before=["index_online", "index_offline", "suggest_tags"],
        should_run_after=["scrape"]
    ),

    # -------------------------------------------------------------------------
    # INDEXING JOBS
    # -------------------------------------------------------------------------

    "index_online": JobSchema(
        job_type="index_online",
        label="Index (Online)",
        description="Create 1536-dim embeddings using OpenAI API for cloud search",
        category="indexing",
        endpoint="/api/create-index",
        weight=40,
        params=[
            JobParam("limit", "int", "Document Limit", default=1000,
                     min_value=1, max_value=100000,
                     description="Max documents to index (0 = unlimited)"),
            JobParam("force_reindex", "bool", "Force Reindex", default=False,
                     description="Reindex even if vectors exist"),
            JobParam("language_filter", "string", "Language Filter", default="",
                     description="ISO code to filter documents"),
            JobParam("resume", "bool", "Resume", default=False,
                     description="Resume from checkpoint if available"),
            # dimension is fixed at 1536 for online
        ],
        should_run_after=["metadata", "clear_vectors"]
    ),

    "index_offline": JobSchema(
        job_type="index_offline",
        label="Index (Offline)",
        description="Create embeddings using local model for offline search",
        category="indexing",
        endpoint="/api/create-index",
        weight=40,
        params=[
            JobParam("limit", "int", "Document Limit", default=1000,
                     min_value=1, max_value=100000,
                     description="Max documents to index (0 = unlimited)"),
            JobParam("force_reindex", "bool", "Force Reindex", default=False,
                     description="Reindex even if vectors exist"),
            JobParam("language_filter", "string", "Language Filter", default="",
                     description="ISO code to filter documents"),
            JobParam("resume", "bool", "Resume", default=False,
                     description="Resume from checkpoint if available"),
            JobParam("dimension", "select", "Dimension", default="768",
                     options=["384", "768", "1024"],
                     description="Embedding dimension (depends on local model)")
        ],
        should_run_after=["metadata", "clear_vectors"]
    ),

    "clear_vectors": JobSchema(
        job_type="clear_vectors",
        label="Clear Vectors",
        description="Remove embeddings from ChromaDB (keeps backup files)",
        category="indexing",
        endpoint="/api/clear-vectors",
        weight=10,
        params=[
            JobParam("dimension", "select", "Dimension", default="768",
                     options=["768", "1536"],
                     description="Which vector DB to clear")
        ],
        should_run_before=["index_online", "index_offline"]
    ),

    # -------------------------------------------------------------------------
    # MAINTENANCE JOBS
    # -------------------------------------------------------------------------

    "suggest_tags": JobSchema(
        job_type="suggest_tags",
        label="Suggest Tags",
        description="AI-powered tag suggestions based on content analysis",
        category="maintenance",
        endpoint="/api/suggest-tags-job",
        weight=15,
        params=[],  # Only needs source_id
        should_run_after=["metadata"]
    ),

    "delete_source": JobSchema(
        job_type="delete_source",
        label="Delete Source",
        description="Remove source from system (backup files and vectors)",
        category="maintenance",
        endpoint="/api/delete",
        weight=10,
        params=[
            JobParam("delete_files", "bool", "Delete Files", default=True,
                     description="Also delete backup files (not just vectors)")
        ],
        conflicts_with=["scrape", "metadata", "index_online", "index_offline",
                        "suggest_tags", "clear_vectors"]  # Can't chain after delete
    ),

    # -------------------------------------------------------------------------
    # ZIM JOBS
    # -------------------------------------------------------------------------

    "zim_inspect": JobSchema(
        job_type="zim_inspect",
        label="ZIM Inspect",
        description="Analyze ZIM file contents and structure",
        category="content",
        endpoint="/api/zim/inspect",
        weight=10,
        requires_source=False,  # Uses zim_path instead
        params=[
            JobParam("zim_path", "string", "ZIM Path", required=True,
                     description="Path to ZIM file"),
            JobParam("scan_limit", "int", "Scan Limit", default=5000,
                     min_value=100, max_value=100000,
                     description="Max articles to scan"),
            JobParam("min_text_length", "int", "Min Text Length", default=50,
                     min_value=0, max_value=1000,
                     description="Minimum text length to include")
        ]
    ),

    # -------------------------------------------------------------------------
    # CLOUD JOBS
    # -------------------------------------------------------------------------

    "install_source": JobSchema(
        job_type="install_source",
        label="Install Source",
        description="Download source from cloud and import to local ChromaDB",
        category="cloud",
        endpoint="/api/install-source",
        weight=50,
        params=[
            JobParam("include_backup", "bool", "Include Backup", default=False,
                     description="Also download HTML backup files"),
            JobParam("sync_mode", "select", "Sync Mode", default="update",
                     options=["update", "replace"],
                     description="'update' adds/merges, 'replace' deletes old vectors first")
        ]
    ),

    "upload": JobSchema(
        job_type="upload",
        label="Upload",
        description="Upload source to cloud storage (R2/S3)",
        category="cloud",
        endpoint="/api/cloud/upload",
        weight=50,
        params=[
            JobParam("include_backup", "bool", "Include Backup", default=False,
                     description="Include HTML backup files in upload")
        ]
    ),

    "pinecone_sync": JobSchema(
        job_type="pinecone_sync",
        label="Pinecone Sync",
        description="Sync 1536-dim vectors to Pinecone cloud index",
        category="cloud",
        endpoint="/api/cloud/pinecone-sync",
        weight=40,
        params=[
            JobParam("dry_run", "bool", "Dry Run", default=False,
                     description="Check what would sync without actually syncing")
        ],
        should_run_after=["index_online"]
    ),

    "visualisation": JobSchema(
        job_type="visualisation",
        label="Visualize",
        description="Generate 3D knowledge map visualization",
        category="cloud",
        endpoint="/api/cloud/visualise",
        weight=20,
        params=[]
    ),
}


# =============================================================================
# PREDEFINED JOB CHAINS (Templates)
# =============================================================================

PREDEFINED_CHAINS: Dict[str, Dict[str, Any]] = {
    "generate_source": {
        "name": "Generate Source",
        "description": "Full source processing: metadata, both indexes, and tag suggestions",
        "resumable": True,
        "phases": [
            {"job_type": "metadata", "weight": 15, "params": {"resume": True}},
            {"job_type": "index_online", "weight": 35, "params": {"resume": True}},
            {"job_type": "index_offline", "weight": 35, "params": {"resume": True}},
            {"job_type": "suggest_tags", "weight": 15, "params": {}},
        ]
    },

    "regenerate_source": {
        "name": "Regenerate Source",
        "description": "Nuclear option: clear everything and rebuild from scratch",
        "resumable": False,
        "phases": [
            {"job_type": "metadata", "weight": 10, "params": {"resume": False}},
            {"job_type": "clear_vectors", "weight": 5, "params": {"dimension": "1536"}},
            {"job_type": "index_online", "weight": 30, "params": {"force_reindex": True}},
            {"job_type": "clear_vectors", "weight": 5, "params": {"dimension": "768"}},
            {"job_type": "index_offline", "weight": 30, "params": {"force_reindex": True}},
            {"job_type": "suggest_tags", "weight": 20, "params": {}},
        ]
    },

    "reindex_online": {
        "name": "Reindex (Online)",
        "description": "Clear and rebuild 1536-dim cloud vectors",
        "resumable": False,
        "phases": [
            {"job_type": "clear_vectors", "weight": 10, "params": {"dimension": "1536"}},
            {"job_type": "index_online", "weight": 90, "params": {"force_reindex": True}},
        ]
    },

    "reindex_offline": {
        "name": "Reindex (Offline)",
        "description": "Clear and rebuild local vectors",
        "resumable": False,
        "phases": [
            {"job_type": "clear_vectors", "weight": 10, "params": {"dimension": "768"}},
            {"job_type": "index_offline", "weight": 90, "params": {"force_reindex": True}},
        ]
    },

    "cloud_publish": {
        "name": "Cloud Publish",
        "description": "Sync to Pinecone and regenerate visualization",
        "resumable": True,
        "phases": [
            {"job_type": "pinecone_sync", "weight": 80, "params": {}},
            {"job_type": "visualisation", "weight": 20, "params": {}},
        ]
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_job_schema(job_type: str) -> Optional[JobSchema]:
    """Get schema for a job type"""
    return JOB_SCHEMAS.get(job_type)


def get_jobs_by_category(category: str) -> List[JobSchema]:
    """Get all jobs in a category"""
    return [j for j in JOB_SCHEMAS.values() if j.category == category]


def get_all_categories() -> List[str]:
    """Get list of all job categories"""
    return list(set(j.category for j in JOB_SCHEMAS.values()))


def validate_job_chain(job_types: List[str]) -> Dict[str, Any]:
    """
    Validate a job chain for ordering issues and conflicts.

    Returns:
        {
            "valid": bool,
            "warnings": [...],  # Non-fatal issues
            "errors": [...]     # Fatal issues that prevent running
        }
    """
    warnings = []
    errors = []

    seen_jobs = set()

    for i, job_type in enumerate(job_types):
        schema = get_job_schema(job_type)
        if not schema:
            errors.append(f"Unknown job type: {job_type}")
            continue

        # Check conflicts
        for conflict in schema.conflicts_with:
            if conflict in seen_jobs:
                errors.append(f"'{schema.label}' conflicts with '{conflict}' - cannot be in same chain")

        # Check ordering - warn if job should run before something we already did
        for should_be_before in schema.should_run_before:
            if should_be_before in seen_jobs:
                warnings.append(
                    f"'{schema.label}' typically runs before '{should_be_before}' - check order"
                )

        # Check ordering - warn if we're running before something we should run after
        for j in range(i + 1, len(job_types)):
            future_job = job_types[j]
            if future_job in schema.should_run_after:
                warnings.append(
                    f"'{schema.label}' typically runs after '{future_job}' - check order"
                )

        seen_jobs.add(job_type)

    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors
    }


def schema_to_dict(schema: JobSchema) -> Dict[str, Any]:
    """Convert JobSchema to JSON-serializable dict for API responses"""
    return {
        "job_type": schema.job_type,
        "label": schema.label,
        "description": schema.description,
        "category": schema.category,
        "weight": schema.weight,
        "requires_source": schema.requires_source,
        "endpoint": schema.endpoint,
        "params": [
            {
                "name": p.name,
                "type": p.type,
                "label": p.label,
                "default": p.default,
                "required": p.required,
                "description": p.description,
                "options": p.options,
                "min_value": p.min_value,
                "max_value": p.max_value
            }
            for p in schema.params
        ],
        "should_run_before": schema.should_run_before,
        "should_run_after": schema.should_run_after,
        "conflicts_with": schema.conflicts_with
    }


def get_all_schemas_dict() -> Dict[str, Any]:
    """Get all schemas as JSON-serializable dict"""
    return {
        "jobs": {k: schema_to_dict(v) for k, v in JOB_SCHEMAS.items()},
        "categories": get_all_categories(),
        "predefined_chains": PREDEFINED_CHAINS
    }
