# Local-development image for the FastAPI service (docker-compose.yml).
# Not a multi-stage/production build -- see docs/architecture.md §2.11 for why
# uv is the project's package manager.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

COPY . .
RUN uv sync --locked

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "api"]
