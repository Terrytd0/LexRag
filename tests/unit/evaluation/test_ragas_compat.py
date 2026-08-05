from __future__ import annotations

from ragas.llms.base import InstructorLLM

from evaluation._ragas_compat import (
    _fixed_map_openai_params,
    _ragas_has_dotted_version_bug,
    ensure_ragas_dotted_version_support,
)


def _probe(model: str) -> InstructorLLM:
    instance = InstructorLLM.__new__(InstructorLLM)
    instance.provider = "openai"
    instance.model = model
    instance.model_args = {"max_tokens": 1024, "temperature": 0.01, "top_p": 0.1}
    return instance


def test_fixed_map_openai_params_remaps_dotted_gpt5_version() -> None:
    mapped = _fixed_map_openai_params(_probe("gpt-5.6-luna"))

    assert "max_tokens" not in mapped
    assert mapped["max_completion_tokens"] == 1024
    assert mapped["temperature"] == 1.0
    assert "top_p" not in mapped


def test_fixed_map_openai_params_leaves_legacy_models_unchanged() -> None:
    mapped = _fixed_map_openai_params(_probe("gpt-4o-mini"))

    assert mapped["max_tokens"] == 1024
    assert "max_completion_tokens" not in mapped
    assert mapped["temperature"] == 0.01


def test_fixed_map_openai_params_still_handles_integer_gpt5() -> None:
    mapped = _fixed_map_openai_params(_probe("gpt-5"))

    assert "max_tokens" not in mapped
    assert mapped["max_completion_tokens"] == 1024


def test_ensure_ragas_dotted_version_support_fixes_the_live_installation() -> None:
    """Whether or not the installed ragas still has #2708, after calling this the
    live `InstructorLLM._map_openai_params` must correctly remap a dotted-version
    reasoning model -- this is the actual guarantee callers depend on, checked
    without asserting anything about *how* it's satisfied (patched vs. already-fixed
    upstream).
    """
    ensure_ragas_dotted_version_support()

    probe = _probe("gpt-5.6-luna")
    mapped = probe._map_provider_params()

    assert "max_tokens" not in mapped
    assert mapped.get("max_completion_tokens") == 1024


def test_ragas_has_dotted_version_bug_returns_a_bool() -> None:
    # Documents the probe's contract without asserting a specific installed-ragas
    # outcome, since that's exactly what varies once ragas ships the upstream fix.
    assert isinstance(_ragas_has_dotted_version_bug(), bool)
