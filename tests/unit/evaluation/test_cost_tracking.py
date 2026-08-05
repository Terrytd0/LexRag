from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

from evaluation.cost_tracking import (
    ModelPricing,
    UsageTracker,
    instrument_async_chat_client,
    instrument_async_embeddings_client,
    instrument_sync_chat_client,
)


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int = 0


def test_usage_tracker_accumulates_across_calls_by_label() -> None:
    tracker = UsageTracker()

    tracker.record("generation", prompt=100, completion=50)
    tracker.record("generation", prompt=200, completion=75)
    tracker.record("judge", prompt=10, completion=5)

    assert tracker.prompt_tokens == {"generation": 300, "judge": 10}
    assert tracker.completion_tokens == {"generation": 125, "judge": 5}
    assert tracker.call_count == {"generation": 2, "judge": 1}
    assert tracker.total_prompt_tokens() == 310
    assert tracker.total_completion_tokens() == 130


def test_usage_tracker_cost_computes_from_pricing() -> None:
    tracker = UsageTracker()
    tracker.record("generation", prompt=1_000_000, completion=1_000_000)
    pricing = ModelPricing(input_per_million=0.20, output_per_million=1.25)

    assert tracker.cost(pricing) == 0.20 + 1.25


def test_usage_tracker_cost_for_labels_restricts_to_given_labels() -> None:
    tracker = UsageTracker()
    tracker.record("generation", prompt=1_000_000, completion=0)
    tracker.record("judge", prompt=1_000_000, completion=0)
    pricing = ModelPricing(input_per_million=1.0, output_per_million=1.0)

    assert tracker.cost_for_labels({"generation"}, pricing) == 1.0
    assert tracker.cost_for_labels({"generation", "judge"}, pricing) == 2.0


async def test_instrument_async_chat_client_records_usage_and_passes_response_through() -> None:
    tracker = UsageTracker()
    client = MagicMock()
    response = MagicMock(usage=_Usage(prompt_tokens=42, completion_tokens=7))
    original_create = AsyncMock(return_value=response)
    client.chat.completions.create = original_create

    instrument_async_chat_client(client, tracker, "judge")
    result = await client.chat.completions.create(model="x", messages=[])

    assert result is response
    assert tracker.prompt_tokens == {"judge": 42}
    assert tracker.completion_tokens == {"judge": 7}
    original_create.assert_awaited_once_with(model="x", messages=[])


async def test_instrument_async_chat_client_handles_missing_usage_gracefully() -> None:
    tracker = UsageTracker()
    client = MagicMock()
    response = MagicMock(usage=None)
    client.chat.completions.create = AsyncMock(return_value=response)

    instrument_async_chat_client(client, tracker, "judge")
    await client.chat.completions.create()

    assert tracker.prompt_tokens == {}


def test_instrument_sync_chat_client_records_usage() -> None:
    tracker = UsageTracker()
    client = MagicMock()
    response = MagicMock(usage=_Usage(prompt_tokens=10, completion_tokens=3))
    client.chat.completions.create = MagicMock(return_value=response)

    instrument_sync_chat_client(client, tracker, "generation")
    result = client.chat.completions.create(model="x", messages=[])

    assert result is response
    assert tracker.prompt_tokens == {"generation": 10}
    assert tracker.completion_tokens == {"generation": 3}


async def test_instrument_async_embeddings_client_records_prompt_tokens_only() -> None:
    tracker = UsageTracker()
    client = MagicMock()
    response = MagicMock(usage=_Usage(prompt_tokens=15))
    client.embeddings.create = AsyncMock(return_value=response)

    instrument_async_embeddings_client(client, tracker, "judge_embedding")
    await client.embeddings.create(input="text", model="text-embedding-3-small")

    assert tracker.prompt_tokens == {"judge_embedding": 15}
    assert tracker.completion_tokens == {"judge_embedding": 0}
