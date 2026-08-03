"""Process entrypoint: `uv run python -m api` starts the server on
APP_HOST/APP_PORT from configs.settings, with reload enabled outside
production. Prefer this over a bare `uvicorn` invocation so host/port stay
config-driven (see .env.example) instead of duplicated as CLI flags.
"""

from __future__ import annotations

import uvicorn

from configs.settings import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.environment != "production",
    )


if __name__ == "__main__":
    main()
