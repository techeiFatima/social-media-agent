# Social Media Agent (RAG)

This repo contains a **local RAG knowledge base** (SQLite + FTS5 + sqlite-vec + `fastembed`) plus small scripts to:
- **Ingest** markdown docs into your knowledge base
- **Retrieve** relevant context with hybrid search
- **Generate** a grounded social post (OpenAI-compatible API)
- Optionally **publish** to Mastodon (if configured)

## File layout (what to implement first)

- `knowledge_base.py`: all retrieval + storage (your “RAG engine”)
- `ingest.py`: CLI to ingest/refresh docs
- `rag_post.py`: CLI to retrieve → generate → (optional) publish
- `config.py`: environment settings
- `llm_client.py`: OpenAI/OpenRouter-compatible generation
- `mastodon_client.py`: publishing helper

## Quickstart

1) Install deps (from your `pyproject.toml`):

```bash
python -m pip install -U pip
python -m pip install -e .
```

2) Put knowledge files in `business-docs/` (create it if needed) as `.md` files.

3) Ingest your docs:

```bash
python ingest.py
```

4) Add a `.env` file (recommended) with:

```bash
OPENAI_API_KEY=your_key_here
# Optional for OpenRouter:
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=gpt-4o-mini
```

5) Generate a post:

```bash
python rag_post.py --topic "AI consulting services" --brand "Emanon" --platform mastodon
```

6) Publish to Mastodon (optional):

```bash
MASTODON_BASE_URL=https://YOUR.INSTANCE
MASTODON_ACCESS_TOKEN=YOUR_TOKEN
python rag_post.py --topic "AI consulting services" --publish
```

