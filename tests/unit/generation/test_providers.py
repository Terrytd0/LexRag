from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from configs.settings import Settings
from generation.providers import OpenAIProvider, get_llm_provider


def test_openai_provider_sends_system_and_user_messages() -> None:
    client = MagicMock(name="openai_client")
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="the answer"))]
    client.chat.completions.create.return_value = response

    provider = OpenAIProvider(client=client, model="gpt-4.1-mini")
    result = provider.complete("system prompt", "user prompt")

    assert result == "the answer"
    client.chat.completions.create.assert_called_once_with(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ],
    )


def test_openai_provider_returns_empty_string_for_null_content() -> None:
    client = MagicMock(name="openai_client")
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=None))]
    client.chat.completions.create.return_value = response

    provider = OpenAIProvider(client=client, model="gpt-4.1-mini")

    assert provider.complete("system", "user") == ""


def test_get_llm_provider_rejects_unsupported_provider() -> None:
    settings = Settings(LLM_PROVIDER="anthropic")

    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        get_llm_provider(settings)


def test_get_llm_provider_returns_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "generation.providers.get_openai_client", lambda: MagicMock(name="openai_client")
    )
    settings = Settings(LLM_PROVIDER="openai", LLM_MODEL="gpt-4.1-mini")

    provider = get_llm_provider(settings)

    assert isinstance(provider, OpenAIProvider)
