"""
pgvector (Supabase) vector store implementation.

This is designed for integration with Supabase's pgvector extension.
Provides async interface for use with Edge Functions or async Python.

Expected table schema:
    CREATE TABLE source_vectors (
        id text PRIMARY KEY,
        source_id text NOT NULL,
        content text NOT NULL,
        embedding vector(768),
        url text,
        local_url text,
        doc_type text,
        metadata jsonb,
        created_at timestamp DEFAULT now()
    );

    CREATE INDEX idx_source_vectors_embedding
    ON source_vectors USING hnsw (embedding vector_cosine_ops);

    CREATE INDEX idx_source_vectors_source_id
    ON source_vectors(source_id);
"""

from typing import List, Dict, Any, Optional, Callable
import os

from .base import VectorStoreBase
from ..schemas import SearchResult, SourceInfo


class PgVectorStore(VectorStoreBase):
    """
    Supabase pgvector implementation.

    Supports both async (asyncpg) and sync (psycopg2) modes.
    Async is preferred for production use.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        table_name: str = "source_vectors",
        embedding_dimension: int = 768,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Initialize pgvector store.

        Args:
            connection_string: PostgreSQL connection string (or from env SUPABASE_DB_URL)
            table_name: Table name for vectors (default: source_vectors)
            embedding_dimension: Dimension of embeddings (must match table schema)
            embedding_function: Optional function to generate embeddings from text.
                               If not provided, uses OpenAI text-embedding-3-small.
        """
        self.connection_string = connection_string or os.getenv("SUPABASE_DB_URL") or os.getenv("PGVECTOR_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError(
                "Connection string required. Set SUPABASE_DB_URL or PGVECTOR_CONNECTION_STRING "
                "environment variable, or pass connection_string parameter."
            )

        self.table_name = table_name
        self.dimension = embedding_dimension
        self._embed = embedding_function
        self._pool = None
        self._sync_conn = None

    async def _get_pool(self):
        """Get or create async connection pool."""
        if self._pool is None:
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=1,
                    max_size=10
                )
            except ImportError:
                raise ImportError(
                    "asyncpg not installed. Install with: pip install asyncpg\n"
                    "Or use sync methods with psycopg2."
                )
        return self._pool

    def _get_sync_conn(self):
        """Get or create sync connection."""
        if self._sync_conn is None:
            try:
                import psycopg2
                self._sync_conn = psycopg2.connect(self.connection_string)
            except ImportError:
                raise ImportError(
                    "psycopg2 not installed. Install with: pip install psycopg2-binary"
                )
        return self._sync_conn

    async def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for query text."""
        if self._embed:
            # Use provided embedding function
            result = self._embed(query)
            # Handle both sync and async functions
            if hasattr(result, "__await__"):
                return await result
            return result

        # Default: use OpenAI
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI()
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=query,
                dimensions=self.dimension
            )
            return response.data[0].embedding
        except ImportError:
            raise ImportError(
                "OpenAI not installed. Install with: pip install openai\n"
                "Or provide an embedding_function parameter."
            )

    def _embed_query_sync(self, query: str) -> List[float]:
        """Synchronous embedding for query text."""
        if self._embed:
            return self._embed(query)

        # Default: use OpenAI
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=query,
                dimensions=self.dimension
            )
            return response.data[0].embedding
        except ImportError:
            raise ImportError(
                "OpenAI not installed. Install with: pip install openai\n"
                "Or provide an embedding_function parameter."
            )

    async def search(
        self,
        query: str,
        n_results: int = 10,
        sources: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Semantic search using pgvector.

        Args:
            query: Natural language search query
            n_results: Maximum number of results
            sources: Optional list of source IDs to filter by

        Returns:
            List of SearchResult objects
        """
        pool = await self._get_pool()
        query_embedding = await self._embed_query(query)

        # Format embedding for PostgreSQL
        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

        async with pool.acquire() as conn:
            if sources:
                # Filter by source IDs
                sql = f"""
                    SELECT id, source_id, content, url,
                           1 - (embedding <=> $1::vector) as similarity,
                           metadata, doc_type
                    FROM {self.table_name}
                    WHERE source_id = ANY($2)
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                """
                rows = await conn.fetch(sql, embedding_str, sources, n_results)
            else:
                # No source filter
                sql = f"""
                    SELECT id, source_id, content, url,
                           1 - (embedding <=> $1::vector) as similarity,
                           metadata, doc_type
                    FROM {self.table_name}
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                """
                rows = await conn.fetch(sql, embedding_str, n_results)

        return [SearchResult.from_pgvector(dict(row)) for row in rows]

    def search_sync(
        self,
        query: str,
        n_results: int = 10,
        sources: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Synchronous version of search."""
        conn = self._get_sync_conn()
        query_embedding = self._embed_query_sync(query)

        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

        with conn.cursor() as cur:
            if sources:
                sql = f"""
                    SELECT id, source_id, content, url,
                           1 - (embedding <=> %s::vector) as similarity,
                           metadata, doc_type
                    FROM {self.table_name}
                    WHERE source_id = ANY(%s)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                cur.execute(sql, (embedding_str, sources, embedding_str, n_results))
            else:
                sql = f"""
                    SELECT id, source_id, content, url,
                           1 - (embedding <=> %s::vector) as similarity,
                           metadata, doc_type
                    FROM {self.table_name}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                cur.execute(sql, (embedding_str, embedding_str, n_results))

            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

        return [
            SearchResult.from_pgvector(dict(zip(columns, row)))
            for row in rows
        ]

    async def get_sources(self) -> List[SourceInfo]:
        """List all available sources with document counts."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            sql = f"""
                SELECT source_id as id, source_id as name, COUNT(*) as count
                FROM {self.table_name}
                GROUP BY source_id
                ORDER BY source_id
            """
            rows = await conn.fetch(sql)

        return [
            SourceInfo(
                id=row["id"],
                name=row["name"],
                count=row["count"]
            )
            for row in rows
        ]

    def get_sources_sync(self) -> List[SourceInfo]:
        """Synchronous version of get_sources."""
        conn = self._get_sync_conn()

        with conn.cursor() as cur:
            sql = f"""
                SELECT source_id as id, source_id as name, COUNT(*) as count
                FROM {self.table_name}
                GROUP BY source_id
                ORDER BY source_id
            """
            cur.execute(sql)
            rows = cur.fetchall()

        return [
            SourceInfo(id=row[0], name=row[1], count=row[2])
            for row in rows
        ]

    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """
        Add documents to the vector store.

        Args:
            documents: List of document dicts with keys:
                - id: Document ID
                - content: Text content
                - source_id: Source identifier
                - url: Document URL
                - metadata: Additional metadata dict
            embeddings: Pre-computed embeddings (optional)

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        pool = await self._get_pool()

        # Compute embeddings if not provided
        if embeddings is None:
            embeddings = []
            for doc in documents:
                emb = await self._embed_query(doc.get("content", ""))
                embeddings.append(emb)

        async with pool.acquire() as conn:
            # Prepare batch insert
            values = []
            for doc, emb in zip(documents, embeddings):
                embedding_str = f"[{','.join(str(x) for x in emb)}]"
                import json
                metadata_json = json.dumps(doc.get("metadata", {}))

                values.append((
                    doc.get("id"),
                    doc.get("source_id"),
                    doc.get("content", ""),
                    embedding_str,
                    doc.get("url", ""),
                    doc.get("local_url", ""),
                    doc.get("doc_type", "article"),
                    metadata_json,
                ))

            # Use executemany for batch insert
            sql = f"""
                INSERT INTO {self.table_name}
                (id, source_id, content, embedding, url, local_url, doc_type, metadata)
                VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    url = EXCLUDED.url,
                    metadata = EXCLUDED.metadata
            """

            await conn.executemany(sql, values)

        return len(documents)

    async def delete_source(self, source_id: str) -> int:
        """Delete all documents from a source."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            sql = f"DELETE FROM {self.table_name} WHERE source_id = $1"
            result = await conn.execute(sql, source_id)
            # Parse "DELETE N" result
            count = int(result.split()[-1]) if result else 0

        return count

    async def health_check(self) -> bool:
        """Check if the database is accessible."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    def close_sync(self):
        """Close sync connection."""
        if self._sync_conn:
            self._sync_conn.close()
            self._sync_conn = None
