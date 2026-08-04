"""Provider-agnostic LLM interface for generation (`docs/architecture.md` §2.9).

`LLMProvider` is the only surface `generation.generator.GenerationService`
depends on -- swapping models or providers is a config change (`LLM_PROVIDER`/
`LLM_MODEL`), not a call-site rewrite.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from openai import OpenAI

from configs.settings import Settings, get_settings


class LLMProvider(Protocol):
    """A single chat-completion call, abstracted away from any specific SDK."""

    def complete(self, system: str, user: str) -> str:
        """Return the model's text completion for a system + user message pair."""
        ...


class OpenAIProvider:
    """`LLMProvider` backed by an OpenAI-compatible chat completions API."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


@lru_cache
def get_openai_client() -> OpenAI:
    """Process-wide OpenAI client singleton, mirroring `ingestion.repository.get_mongo_client`."""
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key or None)


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Construct the `LLMProvider` configured by `Settings.llm_provider`.

    Only `"openai"` is implemented today -- see `docs/architecture.md` §2.9
    for why this stays a thin interface rather than a multi-provider
    framework until a second provider is an actual requirement.
    """
    settings = settings or get_settings()
    if settings.llm_provider != "openai":
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider!r}")
    return OpenAIProvider(client=get_openai_client(), model=settings.llm_model)
