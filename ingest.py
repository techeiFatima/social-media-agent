from __future__ import annotations

import argparse
from pathlib import Path

from config import load_settings
from knowledge_base import KnowledgeBase


def main() -> int:
    settings = load_settings()

    parser = argparse.ArgumentParser(description="Ingest markdown files into the local RAG knowledge base.")
    parser.add_argument("--docs-dir", type=str, default=str(settings.docs_dir), help="Folder containing .md files")
    parser.add_argument("--db-path", type=str, default=str(settings.db_path), help="SQLite DB path")
    parser.add_argument("--no-refresh", action="store_true", help="Do not delete existing chunks before ingesting")
    parser.add_argument("--source-type", type=str, default="business_doc", help="Stored source_type label")
    args = parser.parse_args()

    kb = KnowledgeBase(Path(args.db_path))
    chunks = kb.ingest_markdown_dir(
        Path(args.docs_dir),
        source_type=args.source_type,
        refresh=not args.no_refresh,
    )

    print(f"Ingested {chunks} chunk(s) into {Path(args.db_path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

