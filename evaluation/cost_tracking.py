"""Token-usage/cost instrumentation for the one-off model-comparison run
(`scripts/run_model_comparison.py`) -- not part of the regular evaluation
harness, which tracks no cost today (see `docs/experiments/evaluation_notes.md`
for why: no per-token pricing was verified for the models used until this
comparison specifically needed it).

Wraps a client's `chat.completions.create`/`embeddings.create` to record
`response.usage` as a side effect, then returns the response unchanged. Must be
applied to a **freshly constructed, unwrapped** client -- `instructor` (used by
`ragas.llms.llm_factory`) replaces `chat.completions.create` with its own
structured-output wrapper the first time a client is handed to it, after which
the method's signature no longer matches the raw OpenAI SDK and this
instrumentation can't see `.usage` on the response. Instrument first, then pass
the same (now-wrapped) client into `llm_factory`/`GenerationJudge`/
`OpenAIProvider` -- their wrapping layers call through to this one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI, OpenAI


@dataclass
class ModelPricing:
    """Per-million-token pricing, standard (non-batch) tier, USD."""

    input_per_million: float
    output_per_million: float


@dataclass
class UsageTracker:
    """Accumulates prompt/completion/embedding token counts across every
    instrumented client call, tagged by `label` (e.g. "generation", "judge",
    "judge_embedding") so cost can be broken down by purpose, not just totaled.
    """

    prompt_tokens: dict[str, int] = field(default_factory=dict)
    completion_tokens: dict[str, int] = field(default_factory=dict)
    call_count: dict[str, int] = field(default_factory=dict)

    def record(self, label: str, prompt: int, completion: int) -> None:
        self.prompt_tokens[label] = self.prompt_tokens.get(label, 0) + prompt
        self.completion_tokens[label] = self.completion_tokens.get(label, 0) + completion
        self.call_count[label] = self.call_count.get(label, 0) + 1

    def total_prompt_tokens(self) -> int:
        return sum(self.prompt_tokens.values())

    def total_completion_tokens(self) -> int:
        return sum(self.completion_tokens.values())

    def cost(self, pricing: ModelPricing) -> float:
        """Total USD cost across every recorded call, at `pricing`'s rates.

        Correct only when every recorded label was actually billed at
        `pricing`'s rate -- callers combining multiple models/embedding
        pricing must call this once per pricing tier over the relevant
        subset of labels (see `cost_for_labels`).
        """
        prompt_cost = self.total_prompt_tokens() / 1_000_000 * pricing.input_per_million
        completion_cost = self.total_completion_tokens() / 1_000_000 * pricing.output_per_million
        return prompt_cost + completion_cost

    def cost_for_labels(self, labels: set[str], pricing: ModelPricing) -> float:
        """Cost restricted to the given labels only, at `pricing`'s rates."""
        prompt = sum(self.prompt_tokens.get(label, 0) for label in labels)
        completion = sum(self.completion_tokens.get(label, 0) for label in labels)
        return (prompt / 1_000_000 * pricing.input_per_million) + (
            completion / 1_000_000 * pricing.output_per_million
        )


def instrument_async_chat_client(client: AsyncOpenAI, tracker: UsageTracker, label: str) -> None:
    """Wrap `client.chat.completions.create` (async) to record usage under `label`.

    Must be called before `client` is handed to `ragas.llms.llm_factory` --
    see module docstring.
    """
    original_create = client.chat.completions.create

    async def _tracked_create(*args: Any, **kwargs: Any) -> Any:
        response = await original_create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            tracker.record(label, usage.prompt_tokens, usage.completion_tokens)
        return response

    client.chat.completions.create = _tracked_create  # type: ignore[method-assign]


def instrument_sync_chat_client(client: OpenAI, tracker: UsageTracker, label: str) -> None:
    """Wrap `client.chat.completions.create` (sync) to record usage under `label`."""
    original_create = client.chat.completions.create

    def _tracked_create(*args: Any, **kwargs: Any) -> Any:
        response = original_create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            tracker.record(label, usage.prompt_tokens, usage.completion_tokens)
        return response

    client.chat.completions.create = _tracked_create  # type: ignore[method-assign]


def instrument_async_embeddings_client(
    client: AsyncOpenAI, tracker: UsageTracker, label: str
) -> None:
    """Wrap `client.embeddings.create` (async) to record input-token usage under
    `label`. Embeddings have no completion tokens -- always recorded as 0.
    """
    original_create = client.embeddings.create

    async def _tracked_create(*args: Any, **kwargs: Any) -> Any:
        response = await original_create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            tracker.record(label, usage.prompt_tokens, 0)
        return response

    client.embeddings.create = _tracked_create  # type: ignore[method-assign]
