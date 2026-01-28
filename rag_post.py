from __future__ import annotations

import argparse
from pathlib import Path

from config import load_settings
from knowledge_base import KnowledgeBase
from llm_client import LLMConfig, build_openai_client, generate_social_post
from mastodon_client import MastodonConfig, build_mastodon_client, publish_status


def main() -> int:
    settings = load_settings()

    parser = argparse.ArgumentParser(description="Generate (and optionally publish) a RAG-grounded social post.")
    parser.add_argument("--topic", required=True, help="What the post should be about")
    parser.add_argument("--db-path", type=str, default=str(settings.db_path), help="SQLite DB path")
    parser.add_argument("--top-k", type=int, default=8, help="How many chunks to retrieve")
    parser.add_argument("--keyword-weight", type=float, default=0.5, help="FTS BM25 weight")
    parser.add_argument("--semantic-weight", type=float, default=0.5, help="Vector similarity weight")
    parser.add_argument("--platform", type=str, default="mastodon", help="Platform style label")
    parser.add_argument("--brand", type=str, default="your brand", help="Brand name for tone/voice")
    parser.add_argument("--max-chars", type=int, default=400, help="Character limit for the post")
    parser.add_argument("--hashtags", type=int, default=2, help="Max hashtag count")
    parser.add_argument("--publish", action="store_true", help="Publish to Mastodon if configured")
    parser.add_argument("--visibility", type=str, default="public", help="Mastodon visibility: public|unlisted|private|direct")
    args = parser.parse_args()

    kb = KnowledgeBase(Path(args.db_path))
    results = kb.hybrid_search(
        args.topic,
        keyword_weight=args.keyword_weight,
        semantic_weight=args.semantic_weight,
        top_k=args.top_k,
    )
    context = kb.format_context(results)

    if not settings.openai_api_key:
        print("Missing OPENAI_API_KEY. I can retrieve context, but cannot generate a post yet.\n")
        print("Retrieved context:\n")
        print(context)
        return 2

    client = build_openai_client(
        LLMConfig(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
        )
    )

    post = generate_social_post(
        client=client,
        model=settings.llm_model,
        context=context,
        topic=args.topic,
        platform=args.platform,
        max_chars=args.max_chars,
        hashtags=args.hashtags,
        brand=args.brand,
    )

    print(post)

    if args.publish:
        if not (settings.mastodon_base_url and settings.mastodon_access_token):
            print("\nCannot publish: set MASTODON_BASE_URL and MASTODON_ACCESS_TOKEN in your environment/.env.")
            return 3

        m = build_mastodon_client(
            MastodonConfig(base_url=settings.mastodon_base_url, access_token=settings.mastodon_access_token)
        )
        url = publish_status(m, post, visibility=args.visibility)
        if url:
            print(f"\nPublished: {url}")
        else:
            print("\nPublished.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

