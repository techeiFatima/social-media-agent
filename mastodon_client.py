from __future__ import annotations

from dataclasses import dataclass

from mastodon import Mastodon


@dataclass(frozen=True)
class MastodonConfig:
    base_url: str
    access_token: str


def build_mastodon_client(cfg: MastodonConfig) -> Mastodon:
    return Mastodon(access_token=cfg.access_token, api_base_url=cfg.base_url)


def publish_status(client: Mastodon, text: str, *, visibility: str = "public") -> str:
    """
    Publishes a status and returns the status URL.
    """
    status = client.status_post(text, visibility=visibility)
    # mastodon.py returns dict-like objects; url is commonly present
    url = status.get("url") if isinstance(status, dict) else None
    return str(url or "")

