from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    # RAG / knowledge base
    db_path: Path
    docs_dir: Path

    # LLM (OpenAI-compatible; supports OpenRouter via base_url)
    openai_api_key: str | None
    openai_base_url: str | None
    llm_model: str

    # Mastodon (optional publish)
    mastodon_base_url: str | None
    mastodon_access_token: str | None


def load_settings() -> Settings:
    """
    Loads environment variables from `.env` (if present) and returns Settings.

    Suggested `.env` keys:
    - RAG_DB_PATH=tutorial_rag.db
    - RAG_DOCS_DIR=business-docs
    - OPENAI_API_KEY=...
    - OPENAI_BASE_URL=https://openrouter.ai/api/v1   (optional)
    - LLM_MODEL=nvidia/nemotron-3-nano-30b-a3b:free  (or any OpenAI-compatible model id)
    - MASTODON_BASE_URL=https://mastodon.social
    - MASTODON_ACCESS_TOKEN=...
    """
    load_dotenv()

    root = Path(__file__).resolve().parent

    db_path = Path(os.getenv("RAG_DB_PATH", str(root / "tutorial_rag.db"))).expanduser()
    docs_dir = Path(os.getenv("RAG_DOCS_DIR", str(root / "business-docs"))).expanduser()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL") or None
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    mastodon_base_url = os.getenv("MASTODON_BASE_URL") or None
    mastodon_access_token = os.getenv("MASTODON_ACCESS_TOKEN") or None

    return Settings(
        db_path=db_path,
        docs_dir=docs_dir,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        llm_model=llm_model,
        mastodon_base_url=mastodon_base_url,
        mastodon_access_token=mastodon_access_token,
    )

