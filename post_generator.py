"""
Backwards-compatible wrapper.

Use `rag_post.py` for the CLI, or import `KnowledgeBase` + `llm_client.generate_social_post`
for programmatic usage.
"""

from __future__ import annotations

from pathlib import Path

from config import load_settings
from knowledge_base import KnowledgeBase
from llm_client import LLMConfig, build_openai_client, generate_social_post


def generate_post(topic: str, *, db_path: str | Path | None = None) -> str:
    settings = load_settings()
    kb = KnowledgeBase(Path(db_path) if db_path else settings.db_path)
    results = kb.hybrid_search(topic, top_k=8)
    context = kb.format_context(results)

    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    client = build_openai_client(
        LLMConfig(api_key=settings.openai_api_key, base_url=settings.openai_base_url, model=settings.llm_model)
    )
    return generate_social_post(
        client=client,
        model=settings.llm_model,
        context=context,
        topic=topic,
        brand="your brand",
    )

