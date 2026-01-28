from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    base_url: str | None = None


def build_openai_client(cfg: LLMConfig) -> OpenAI:
    if cfg.base_url:
        return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    return OpenAI(api_key=cfg.api_key)


def generate_social_post(
    *,
    client: OpenAI,
    model: str,
    context: str,
    topic: str,
    platform: str = "mastodon",
    max_chars: int = 400,
    hashtags: int = 2,
    brand: str = "your brand",
) -> str:
    """
    OpenAI-compatible chat completion to generate a grounded social post.
    """
    system = (
        f"You are a social media manager for {brand}. "
        f"Write a {platform} post that is concise (under {max_chars} characters), "
        f"includes {max(0, hashtags-1)}-{hashtags} relevant hashtags, "
        "sounds authentic, and is useful. "
        "Use ONLY information from the provided context. "
        "If the context is insufficient, say you don't have enough info yet and ask a specific follow-up question."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Context:\n\n{context}\n\nCreate a post about: {topic}\n\nOutput only the post text.",
            },
        ],
    )

    return (resp.choices[0].message.content or "").strip()

