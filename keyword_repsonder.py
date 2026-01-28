"""
Example placeholder for a keyword-triggered response flow.

You can later wire this up to Mastodon mentions, Telegram messages, etc.
For now it exposes a `respond()` function using the same RAG knowledge base.
"""

from __future__ import annotations

from pathlib import Path

from config import load_settings
from knowledge_base import KnowledgeBase
from llm_client import LLMConfig, build_openai_client


def respond(message: str, *, db_path: str | Path | None = None) -> str:
    """
    Retrieves context for `message` and returns a short grounded response.
    """
    settings = load_settings()
    kb = KnowledgeBase(Path(db_path) if db_path else settings.db_path)
    results = kb.hybrid_search(message, top_k=6)
    context = kb.format_context(results, max_chars=2500)

    if not settings.openai_api_key:
        return "I can’t respond yet (missing OPENAI_API_KEY), but I did retrieve relevant context locally."

    client = build_openai_client(LLMConfig(api_key=settings.openai_api_key, base_url=settings.openai_base_url, model=settings.llm_model))

    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Use ONLY the provided context. Keep the response under 500 characters.",
            },
            {"role": "user", "content": f"Context:\n\n{context}\n\nMessage:\n{message}\n\nReply:"},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

