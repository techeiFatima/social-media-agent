from __future__ import annotations

import json
import os
import re
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import sqlite_vec
from fastembed import TextEmbedding


@dataclass(frozen=True)
class SearchResult:
    id: int
    content: str
    source_type: str
    source_id: str
    metadata: dict[str, Any]
    bm25_score: float
    semantic_score: float
    final_score: float


class KnowledgeBase:
    """
    Local RAG knowledge base:
    - Stores content + metadata in SQLite
    - Keyword search via FTS5 (BM25)
    - Semantic search via sqlite-vec (cosine distance)
    - Hybrid scoring = weighted BM25 + weighted semantic
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

        # Make embedding deterministic, local and warning-free
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        self._embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self._conn: sqlite3.Connection | None = None
        self._init_database()

    # ---------- DB ----------
    def connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is None:
            return
        self._conn.close()
        self._conn = None

    def _init_database(self) -> None:
        conn = self.connect()
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                source_id TEXT,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # 384 dims for all-MiniLM-L6-v2
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                embedding float[384] distance_metric=cosine
            )
            """
        )

        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_fts USING fts5(
                content,
                source_type,
                source_id,
                content='embeddings_meta',
                content_rowid='id'
            )
            """
        )

        # Keep FTS in sync with meta table
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS embeddings_ai AFTER INSERT ON embeddings_meta BEGIN
                INSERT INTO embeddings_fts(rowid, content, source_type, source_id)
                VALUES (new.id, new.content, new.source_type, new.source_id);
            END
            """
        )

        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS embeddings_ad AFTER DELETE ON embeddings_meta BEGIN
                INSERT INTO embeddings_fts(embeddings_fts, rowid, content, source_type, source_id)
                VALUES ('delete', old.id, old.content, old.source_type, old.source_id);
            END
            """
        )

        conn.commit()

    # ---------- Embeddings / chunking ----------
    def chunk_markdown_by_h2(self, content: str, filename: str) -> list[dict[str, Any]]:
        """
        Chunk a markdown doc by `##` headers, while keeping the doc title.
        Returns list of {"content": str, "metadata": {...}}.
        """
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        doc_title = title_match.group(1) if title_match else filename

        sections = re.split(r"(?=^##\s+)", content, flags=re.MULTILINE)
        chunks: list[dict[str, Any]] = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            section_title_match = re.search(r"^##\s+(.+)$", section, re.MULTILINE)
            section_title = section_title_match.group(1) if section_title_match else "Introduction"

            chunk_content = f"[From: {filename}]\n# {doc_title}\n\n{section}"
            chunks.append(
                {
                    "content": chunk_content,
                    "metadata": {"source_file": filename, "section_title": section_title},
                }
            )

        return chunks if chunks else [{"content": content, "metadata": {"source_file": filename}}]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = list(self._embedder.embed(texts))
        return [emb.tolist() for emb in embeddings]

    @staticmethod
    def _serialize_embedding(embedding: list[float]) -> bytes:
        return struct.pack(f"{len(embedding)}f", *embedding)

    def _save_embedding(
        self,
        *,
        source_type: str,
        content: str,
        embedding: list[float],
        source_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> int:
        conn = self.connect()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO embeddings_meta (source_type, source_id, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                source_type,
                source_id,
                content,
                json.dumps(metadata) if metadata else None,
                datetime.now().isoformat(),
            ),
        )
        rowid = int(cur.lastrowid)

        cur.execute(
            """
            INSERT INTO vec_embeddings (rowid, embedding)
            VALUES (?, ?)
            """,
            (rowid, self._serialize_embedding(embedding)),
        )

        conn.commit()
        return rowid

    # ---------- Ingest ----------
    def delete_source(self, *, source_type: str, source_id: str) -> int:
        """
        Deletes all chunks for a given (source_type, source_id). Returns deleted row count.
        """
        conn = self.connect()
        cur = conn.cursor()

        cur.execute(
            "SELECT id FROM embeddings_meta WHERE source_type = ? AND source_id = ?",
            (source_type, source_id),
        )
        ids = [int(r[0]) for r in cur.fetchall()]
        if not ids:
            return 0

        placeholders = ",".join("?" * len(ids))
        cur.execute(f"DELETE FROM embeddings_meta WHERE id IN ({placeholders})", ids)
        # vec_embeddings is linked by rowid, so delete those too
        cur.execute(f"DELETE FROM vec_embeddings WHERE rowid IN ({placeholders})", ids)
        conn.commit()
        return len(ids)

    def ingest_markdown_dir(
        self,
        docs_dir: Path,
        *,
        source_type: str = "business_doc",
        refresh: bool = True,
        glob_pattern: str = "*.md",
    ) -> int:
        """
        Ingest all markdown files in `docs_dir`. Returns number of chunks stored.
        If refresh=True, deletes existing chunks for the same file before re-ingesting.
        """
        docs_dir = Path(docs_dir)
        docs_dir.mkdir(parents=True, exist_ok=True)

        md_files = sorted(docs_dir.glob(glob_pattern))
        total_chunks = 0

        for path in md_files:
            content = path.read_text(encoding="utf-8")
            if refresh:
                self.delete_source(source_type=source_type, source_id=path.name)

            chunks = self.chunk_markdown_by_h2(content, path.name)
            texts = [c["content"] for c in chunks]
            embeddings = self.embed_texts(texts)

            for chunk, emb in zip(chunks, embeddings):
                self._save_embedding(
                    source_type=source_type,
                    content=chunk["content"],
                    embedding=emb,
                    source_id=path.name,
                    metadata=chunk.get("metadata") or {},
                )

            total_chunks += len(chunks)

        return total_chunks

    # ---------- Search ----------
    def bm25_search(self, query: str, *, limit: int = 100) -> dict[int, float]:
        conn = self.connect()
        cur = conn.cursor()
        safe_query = query.replace('"', '""')
        try:
            cur.execute(
                """
                SELECT rowid, bm25(embeddings_fts) as score
                FROM embeddings_fts
                WHERE embeddings_fts MATCH ?
                LIMIT ?
                """,
                (safe_query, limit),
            )
            return {int(row[0]): float(row[1]) for row in cur.fetchall()}
        except sqlite3.OperationalError:
            return {}

    def semantic_search(self, query_embedding: list[float], *, limit: int = 100) -> dict[int, float]:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT rowid, distance
            FROM vec_embeddings
            WHERE embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (self._serialize_embedding(query_embedding), limit),
        )
        return {int(row[0]): float(row[1]) for row in cur.fetchall()}

    @staticmethod
    def _normalize_bm25(bm25_scores: dict[int, float]) -> dict[int, float]:
        if not bm25_scores:
            return {}
        scores = list(bm25_scores.values())
        min_score = min(scores)  # most negative = best
        max_score = max(scores)
        if min_score == max_score:
            return {i: 1.0 for i in bm25_scores}
        rng = max_score - min_score
        return {i: (max_score - s) / rng for i, s in bm25_scores.items()}

    @staticmethod
    def _normalize_distances(distances: dict[int, float]) -> dict[int, float]:
        if not distances:
            return {}
        similarities = {i: 1.0 - (d / 2.0) for i, d in distances.items()}
        min_sim = min(similarities.values())
        max_sim = max(similarities.values())
        if min_sim == max_sim:
            return {i: 1.0 for i in similarities}
        rng = max_sim - min_sim
        return {i: (s - min_sim) / rng for i, s in similarities.items()}

    def _get_meta_by_ids(self, ids: Iterable[int]) -> dict[int, dict[str, Any]]:
        ids = list(ids)
        if not ids:
            return {}

        conn = self.connect()
        cur = conn.cursor()
        placeholders = ",".join("?" * len(ids))
        cur.execute(
            f"""
            SELECT id, source_type, source_id, content, metadata
            FROM embeddings_meta
            WHERE id IN ({placeholders})
            """,
            ids,
        )

        out: dict[int, dict[str, Any]] = {}
        for row in cur.fetchall():
            out[int(row[0])] = {
                "source_type": str(row[1]),
                "source_id": str(row[2] or ""),
                "content": str(row[3]),
                "metadata": json.loads(row[4]) if row[4] else {},
            }
        return out

    def hybrid_search(
        self,
        query: str,
        *,
        keyword_weight: float = 0.5,
        semantic_weight: float = 0.5,
        top_k: int = 10,
        candidate_k: int = 100,
    ) -> list[SearchResult]:
        query_embedding = self.embed_texts([query])[0]

        bm25_raw = self.bm25_search(query, limit=candidate_k)
        bm25_norm = self._normalize_bm25(bm25_raw)

        sem_raw = self.semantic_search(query_embedding, limit=candidate_k)
        sem_norm = self._normalize_distances(sem_raw)

        all_ids = set(bm25_norm.keys()) | set(sem_norm.keys())
        if not all_ids:
            return []

        meta = self._get_meta_by_ids(all_ids)
        results: list[SearchResult] = []

        for i in all_ids:
            b = bm25_norm.get(i, 0.0)
            s = sem_norm.get(i, 0.0)
            final = (keyword_weight * b) + (semantic_weight * s)
            m = meta.get(i, {})
            results.append(
                SearchResult(
                    id=i,
                    content=m.get("content", ""),
                    source_type=m.get("source_type", ""),
                    source_id=m.get("source_id", ""),
                    metadata=m.get("metadata", {}),
                    bm25_score=b,
                    semantic_score=s,
                    final_score=final,
                )
            )

        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]

    # ---------- RAG formatting ----------
    @staticmethod
    def format_context(results: list[SearchResult], *, max_chars: int = 4000) -> str:
        if not results:
            return "No relevant context found."

        parts: list[str] = []
        chars_used = 0

        for idx, r in enumerate(results, 1):
            header = f"[{idx}. {r.source_type}] (score: {r.final_score:.2f})"
            content = r.content

            available = max_chars - chars_used - len(header) - 10
            if available <= 100:
                break

            if len(content) > available:
                content = content[: available - 3] + "..."

            entry = f"{header}\n{content}\n"
            parts.append(entry)
            chars_used += len(entry)

        return "\n".join(parts)

