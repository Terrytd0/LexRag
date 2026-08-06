# ADR 001: Cross-Encoder Reranker Inference Backend

**Status:** Decided (Sprint 5 Day 6) — keep `RERANK_BACKEND=torch` (the
pre-existing implicit default), reject `onnx` as the new default.

## Context

`docs/experiments/evaluation_notes.md`'s Day 5 baseline confirmed the
cross-encoder reranker (`BAAI/bge-reranker-v2-m3`, §2.7 of
`docs/architecture.md`) as ~95% of end-to-end query latency (53.468s of
56.370s avg) on this machine, which has no GPU/DirectML/OpenVINO
acceleration path (`torch.cuda.is_available()` is `False`; confirmed again
Day 6). `docs/experiments/retrieval_debugging.md`'s Day 4 "Next steps"
explicitly named ONNX/OpenVINO CPU acceleration as an unexplored, orthogonal
lever, separate from the `RERANK_INPUT_TOP_K` candidate-count trim already
applied.

`sentence-transformers` 5.6.1 (installed) supports loading the same
`CrossEncoder` model weights through a `backend` parameter (`"torch"`,
`"onnx"`, or `"openvino"`) with no change to the model itself or to
`CrossEncoderReranker`'s public interface — a backend swap is an inference
runtime choice, not an architectural change to the retrieval/rerank
pipeline, its ADR (§2.7), or any stored data shape.

## Decision

Added `Settings.rerank_backend` (default `"torch"`, threaded through
`CrossEncoderReranker.__init__` to `CrossEncoder(model_name, backend=...)`)
and measured `"onnx"` against the real 30-case golden set
(`scripts/rerank_backend_benchmark.py`), retrieval-only (no LLM cost):

| Backend | Total reranker latency (30 cases, `RERANK_INPUT_TOP_K=20`) | Avg/case |
|---|---|---|
| torch | 1451.202s | 48.373s |
| onnx (dynamic export, default opset, fp32) | 1423.531s | 47.451s |

**Speedup: 1.02x — not a measurable improvement.** For comparison,
`max |rerank_score| delta` across every case/chunk was `0.000002` and
top-1 chunk agreement was 30/30 — the ONNX export is numerically faithful
to the torch model, so the lack of speedup isn't an export-correctness
problem. The default `optimum`-driven ONNX export used here applies no
graph optimization or quantization pass; PyTorch's own CPU backend already
runs this model size with comparable efficiency on this hardware without
either.

**Decision: keep the `torch` backend as the default.** Per CLAUDE.md
("Don't increase architectural complexity unless it improves measurable
retrieval or generation quality" — the same principle extends to latency,
this sprint's other measured axis) and the AI engineering principle
"measure before optimizing," adding `optimum`/`onnxruntime` as a
dependency is not justified by a 1.02x speedup. `Settings.rerank_backend`
is left in place (not reverted) as a documented, tested, zero-cost escape
hatch — `RERANK_BACKEND=onnx` still works and produces consistent scores —
in case a future graph-optimized or quantized export changes this
calculus; today's default stays `torch`.

## Consequences

- No latency win from this lever. `docs/experiments/evaluation_notes_day6.md`
  records the successful latency lever taken instead
  (`RERANK_INPUT_TOP_K` reduction, validated against real golden-set ground
  truth rather than the single-query heuristic Day 4 used).
- `optimum[onnxruntime]` remains an installed dependency
  (`pyproject.toml`) even though the default doesn't use it, so
  `RERANK_BACKEND=onnx` keeps working without a fresh `uv add`. If a later
  session revisits this (e.g. INT8 dynamic quantization via
  `optimum.onnxruntime.ORTQuantizer`, a materially different and riskier
  lever that would need its own measured validation of score drift), this
  dependency is already in place.
- No change to `docs/architecture.md` §2.7 — the reranker model, its role
  in the pipeline, and `CrossEncoderReranker`'s interface are all
  unchanged.
