"""Prepare and publish a hosted-only skating reference in the official cloud index.

This does not upload the copyrighted PDF or create a downloadable source pack.
Run from the public repository root. Generated vectors stay under clippy-core/build/.
"""

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ID = "usfs-rulebook-2026-27"
PDF_SHA256 = "b701a4f83527a394b4ef5486198dc52c2feb11232a9d7b463e83602d388c83f3"
PDF_URL = (
    "https://dxbhsrqyrr690.cloudfront.net/sidearm.nextgen.sites/"
    "usafs.sidearmsports.com/documents/2026/8/13/2026-27_Rulebook.pdf"
)
INDEX = ROOT / "clippy-core" / "build" / "usfs-2026-27.sqlite"
VECTORS = ROOT / "clippy-core" / "build" / "usfs-hosted-vectors.jsonl"
MODEL = "text-embedding-3-small"


def load_source_rows(index_path: Path = INDEX) -> list[dict]:
    with sqlite3.connect(index_path) as db:
        rows = db.execute(
            "SELECT id, content, url, metadata FROM chunks ORDER BY rowid"
        ).fetchall()
    if not rows:
        raise ValueError("The local skating index is empty")

    documents = []
    for chunk_id, content, url, raw_metadata in rows:
        metadata = json.loads(raw_metadata)
        if not content.strip() or not url.startswith(PDF_URL + "#page="):
            raise ValueError(f"Invalid content or publisher citation for {chunk_id}")
        section = str(metadata.get("section_id", "") or "").strip()
        page = int(metadata["page_start"])
        documents.append({
            "id": f"{SOURCE_ID}:{chunk_id}",
            "content": content,
            "metadata": {
                "source": SOURCE_ID,
                "title": f"{section} | 2026-27 U.S. Figure Skating Rulebook" if section else "2026-27 U.S. Figure Skating Rulebook",
                "url": url,
                "content_preview": content[:1000],
                "doc_type": "research",
                "section_id": section,
                "page": page,
                "edition": "2026-27",
                "part": str(metadata.get("part", "")),
                "discipline": str(metadata.get("discipline", "")),
                "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            },
        })
    if len({doc["id"] for doc in documents}) != len(documents):
        raise ValueError("Duplicate hosted vector IDs")
    return documents


def verify_pdf(pdf_path: Path) -> None:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != PDF_SHA256:
        raise ValueError("PDF SHA256 differs from the inspected 2026-27 edition")


def read_prepared_vectors(path: Path = VECTORS) -> dict[str, dict]:
    if not path.exists():
        return {}
    vectors = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["id"] in vectors:
                raise ValueError(f"Duplicate prepared vector: {record['id']}")
            vectors[record["id"]] = record
    return vectors


def prepare(documents: list[dict], pdf_path: Path, batch_size: int = 50) -> None:
    from openai import OpenAI

    verify_pdf(pdf_path)
    VECTORS.parent.mkdir(parents=True, exist_ok=True)
    prepared = read_prepared_vectors()
    expected_ids = {doc["id"] for doc in documents}
    if set(prepared) - expected_ids:
        raise ValueError("Prepared vector file contains IDs from a different index")
    for doc in documents:
        existing = prepared.get(doc["id"])
        if existing and existing["metadata"].get("content_hash") != doc["metadata"]["content_hash"]:
            raise ValueError(f"Prepared vector is stale: {doc['id']}")

    pending = [doc for doc in documents if doc["id"] not in prepared]
    print(f"Preparing {len(pending)} of {len(documents)} hosted vectors")
    if not pending:
        return
    client = OpenAI()
    with VECTORS.open("a", encoding="utf-8") as output:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            response = client.embeddings.create(model=MODEL, input=[doc["content"] for doc in batch])
            embeddings = sorted(response.data, key=lambda item: item.index)
            if len(embeddings) != len(batch):
                raise ValueError("OpenAI returned an incomplete embedding batch")
            for doc, embedding in zip(batch, embeddings):
                if len(embedding.embedding) != 1536:
                    raise ValueError(f"Unexpected embedding dimension for {doc['id']}")
                record = {"id": doc["id"], "values": embedding.embedding, "metadata": doc["metadata"]}
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"Prepared {min(offset + batch_size, len(pending))}/{len(pending)} remaining vectors")


def publish(documents: list[dict], batch_size: int = 50) -> None:
    import boto3
    from pinecone import Pinecone

    vectors = read_prepared_vectors()
    expected_ids = {doc["id"] for doc in documents}
    if set(vectors) != expected_ids:
        raise ValueError(f"Prepared vectors incomplete: {len(vectors)}/{len(expected_ids)}")
    for doc in documents:
        if vectors[doc["id"]]["metadata"].get("content_hash") != doc["metadata"]["content_hash"]:
            raise ValueError(f"Prepared vector is stale: {doc['id']}")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]
    if index_name not in {item.name for item in pc.list_indexes()}:
        raise ValueError(f"Configured Pinecone index does not exist: {index_name}")
    index = pc.Index(index_name)
    if index.describe_index_stats().dimension != 1536:
        raise ValueError("Configured Pinecone index is not 1536-dimensional")

    ordered_vectors = [
        {"id": doc["id"], "values": vectors[doc["id"]]["values"], "metadata": doc["metadata"]}
        for doc in documents
    ]
    for offset in range(0, len(ordered_vectors), batch_size):
        batch = ordered_vectors[offset:offset + batch_size]
        index.upsert(vectors=batch, namespace="default")
        print(f"Uploaded {min(offset + batch_size, len(ordered_vectors))}/{len(ordered_vectors)} vectors")

    sample_ids = [ordered_vectors[i]["id"] for i in (0, len(ordered_vectors) // 2, len(ordered_vectors) - 1)]
    sample = index.fetch(ids=sample_ids, namespace="default")
    if set(sample.vectors) != set(sample_ids):
        raise RuntimeError("Pinecone sample verification failed; catalog was not changed")

    bucket = os.environ["R2_BUCKET_NAME"]
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    key = "published/catalog.json"
    catalog_bytes = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    catalog = json.loads(catalog_bytes)
    sources = catalog["sources"]
    if any(item.get("source_id") == SOURCE_ID for item in sources):
        print("Hosted skating source is already in the catalog")
        return
    (VECTORS.parent / "catalog-before-skating.json").write_bytes(catalog_bytes)
    sources.append({
        "source_id": SOURCE_ID,
        "name": "U.S. Figure Skating Rulebook 2026-27 (demo)",
        "short_name": "Figure skating",
        "description": "Temporary searchable reference to the publisher's 2026-27 rulebook. Results link to the official PDF; this hosted demo does not distribute a PDF or downloadable pack.",
        "license": "Copyright © 2026 U.S. Figure Skating",
        "license_verified": False,
        "tags": ["figure-skating", "rules", "reference"],
        "topics": ["figure-skating", "rules"],
        "base_url": PDF_URL,
        "live_url": "https://usfigureskating.org/sports/2025/8/9/rulebook-bylaws.aspx",
        "backup_url": "",
        "source_type": "pdf-reference",
        "availability": "hosted",
        "reference_only": True,
        "language": "en",
        "doc_count": len(documents),
        "size_bytes": 0,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    })
    catalog["source_count"] = len(sources)
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat()
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(catalog, indent=2, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"Published {SOURCE_ID} to the official catalog")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "prepare", "publish"))
    parser.add_argument("--pdf", type=Path, default=Path(r"E:\Downloads\2026-27_Rulebook.pdf"))
    parser.add_argument("--index", type=Path, default=INDEX)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    documents = load_source_rows(args.index)
    print(f"Source: {SOURCE_ID}; indexed chunks: {len(documents)}; model: {MODEL}")
    if args.action == "prepare":
        prepare(documents, args.pdf)
    elif args.action == "publish":
        verify_pdf(args.pdf)
        publish(documents)


if __name__ == "__main__":
    main()
