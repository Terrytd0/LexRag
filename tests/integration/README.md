# Integration Tests

Exercise real MongoDB / Qdrant / Elasticsearch dependencies (brought up via the
root `docker-compose.yml`, added Day 2). Each test skips itself — via a
`pytest.mark.integration` marker plus a live connectivity check in its fixture —
if the service it needs isn't reachable, so a plain `pytest` run from the repo
root is always safe regardless of what's running locally.

Run explicitly, with the stack up:

```
docker compose up -d mongo qdrant elasticsearch
uv run pytest tests/integration -m integration
```
