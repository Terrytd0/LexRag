"""Compatibility shim for a confirmed upstream ragas bug: dotted minor-version
OpenAI model names (e.g. "gpt-5.6-luna") aren't recognized by
`InstructorLLM._map_openai_params` as "reasoning models" that require
`max_completion_tokens` instead of `max_tokens`, because that method parses the
version number with `int()`, which raises (and silently swallows) `ValueError` on
a string like `"5.6"`. The unmapped `max_tokens` is then sent to OpenAI, which
rejects it with `Unsupported parameter: 'max_tokens'`.

Tracked upstream: https://github.com/vibrantlabsai/ragas/issues/2708 (open as of
2026-08-05). Fix (unmerged): https://github.com/vibrantlabsai/ragas/pull/2725,
whose core change is `int(version_str)` -> `float(version_str)`. This module
applies that same one-line fix locally -- but only if a live functional probe
shows the installed ragas is still affected, so it self-disables the moment ragas
ships the real fix rather than relying on a hardcoded "fixed in version X" check
that could silently go stale.

TODO(ragas#2708): delete this module (and its call site in
`evaluation.metrics.generation.GenerationJudge`) once ragas releases the fix,
and drop the now-unneeded compatibility test in
`tests/unit/evaluation/test_ragas_compat.py`.
"""

from __future__ import annotations

import re
from typing import Any

from ragas.llms.base import InstructorLLM

_PROBE_MODEL = "gpt-5.6-compat-probe"
_checked = False


def _ragas_has_dotted_version_bug() -> bool:
    """Functionally probe the installed ragas for ragas#2708, rather than
    comparing `ragas.__version__` against a hardcoded "fixed" release -- this
    keeps working correctly across ragas upgrades without maintenance.

    Builds a minimal `InstructorLLM` (via `__new__`, skipping `__init__` since
    that requires a real client) with a dotted-version OpenAI model name and
    checks whether `max_tokens` survives parameter mapping unmapped -- which is
    exactly the bug's observable symptom.
    """
    probe = InstructorLLM.__new__(InstructorLLM)
    probe.provider = "openai"
    probe.model = _PROBE_MODEL
    probe.model_args = {"max_tokens": 1024, "temperature": 0.01, "top_p": 0.1}
    mapped = probe._map_provider_params()
    return "max_tokens" in mapped


def _fixed_map_openai_params(self: InstructorLLM) -> dict[str, Any]:
    """Reimplementation of `InstructorLLM._map_openai_params`, differing only in
    parsing the GPT version with `float()` instead of `int()` (PR #2725's fix)
    so dotted minor versions like `"5.6"` are recognized as reasoning models.
    """
    mapped_args = dict(self.model_args)
    model_lower = self.model.lower()

    def is_reasoning_model(model_str: str) -> bool:
        if re.match(r"^o[1-9]\d*(-|_|$)", model_str):
            return True
        if model_str.startswith("gpt-"):
            version_str = model_str[4:].split("-")[0].split("_")[0]
            try:
                if float(version_str) >= 5:
                    return True
            except ValueError:
                pass
        return model_str == "codex-mini"

    requires_max_completion_tokens = is_reasoning_model(model_lower)
    if requires_max_completion_tokens and "max_tokens" in mapped_args:
        mapped_args["max_completion_tokens"] = mapped_args.pop("max_tokens")
    if requires_max_completion_tokens:
        mapped_args["temperature"] = 1.0
        mapped_args.pop("top_p", None)
    return mapped_args


def ensure_ragas_dotted_version_support() -> None:
    """Patch `InstructorLLM._map_openai_params` if (and only if) the installed
    ragas still exhibits ragas#2708. Idempotent -- cheap to call from every
    `GenerationJudge` construction.
    """
    global _checked
    if _checked:
        return
    if _ragas_has_dotted_version_bug():
        InstructorLLM._map_openai_params = _fixed_map_openai_params  # type: ignore[method-assign]
    _checked = True
