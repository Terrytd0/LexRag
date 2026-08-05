"""RAGAS-based generation-quality metrics (faithfulness, context precision/recall,
answer relevancy) via LLM-as-judge (`docs/architecture.md` §2.10), plus a DeepEval
wrapper (`PrecomputedScoreMetric`) that turns each RAGAS score into a
threshold/pass-fail assertion. Per the ADR, DeepEval's role here is the
CI/test-runner integration layer around RAGAS's scores -- not a second, redundant
LLM-judge pass over the same four metrics.

Assumption (documented per the sprint brief, see `docs/experiments/evaluation_notes.md`):
the RAGAS judge and the embedding model used for Answer Relevancy both call OpenAI
(`Settings.llm_model` and `JUDGE_EMBEDDING_MODEL` respectively) -- a cost and
dependency distinct from the system's own local dense-retrieval embeddings
(`Settings.embedding_model`, bge-m3), which never leave the machine. Judge
temperature is left at the SDK default (0 is not independently configurable through
`llm_factory` without a custom client); RAGAS's LLM-judge scores are therefore not
perfectly deterministic run-to-run, per the ADR's stated consequence.
"""

from __future__ import annotations

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from configs.settings import Settings, get_settings
from evaluation._ragas_compat import ensure_ragas_dotted_version_support

JUDGE_EMBEDDING_MODEL = "text-embedding-3-small"

# faithfulness mirrors the acceptance criterion in docs/01-requirements.md §7.4.
# The other three have no documented contractual bar yet -- 0.70 is an
# evaluation-only placeholder pending Day 6 calibration against real runs.
GENERATION_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.90,
    "context_precision": 0.70,
    "context_recall": 0.70,
    "answer_relevancy": 0.70,
}


class GenerationJudge:
    """Owns the RAGAS LLM-judge/embedding clients and the four metric instances --
    constructed once per evaluation run so every case is scored by the same judge.
    """

    def __init__(self, settings: Settings | None = None, client: AsyncOpenAI | None = None) -> None:
        settings = settings or get_settings()
        ensure_ragas_dotted_version_support()
        # RAGAS's `.ascore()` methods require an async-capable client -- the
        # sync `generation.providers.get_openai_client()` singleton (used for
        # the system's own generation calls) raises `TypeError` if handed to
        # `llm_factory` here, since `agenerate()` checks for it explicitly.
        #
        # `client`, if given, is used as-is instead of constructing a fresh one --
        # lets a caller pass a client it has already instrumented (e.g. usage/cost
        # tracking in `evaluation/cost_tracking.py`) before `llm_factory` below
        # wraps it for structured output. Existing callers that don't pass this
        # get identical behaviour to before.
        client = client or AsyncOpenAI(api_key=settings.openai_api_key or None)
        # ragas's InstructorModelArgs defaults max_tokens=1024. Reasoning models
        # (gpt-5.x and o-series) spend part of that budget on internal reasoning
        # before ever emitting the structured-output content, so 1024 truncates
        # mid-response over real (multi-paragraph) retrieved context --
        # `instructor.v2.core.errors.IncompleteOutputException`. ragas's own
        # `InstructorModelArgs` docstring recommends 4096+ for exactly this case.
        llm = llm_factory(settings.llm_model, client=client, max_tokens=4096)
        embeddings = OpenAIEmbeddings(client=client, model=JUDGE_EMBEDDING_MODEL)
        self.faithfulness = Faithfulness(llm=llm)
        self.context_precision = ContextPrecisionWithReference(llm=llm)
        self.context_recall = ContextRecall(llm=llm)
        self.answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)


class GenerationCaseMetrics(BaseModel):
    """RAGAS scores for one answerable, non-refused golden case."""

    case_id: str
    faithfulness: float
    context_precision: float
    context_recall: float
    answer_relevancy: float


async def score_generation_case(
    judge: GenerationJudge,
    case_id: str,
    question: str,
    answer: str,
    reference: str,
    contexts: list[str],
) -> GenerationCaseMetrics:
    """Score one generated answer against its retrieved contexts and reference answer."""
    faithfulness = await judge.faithfulness.ascore(
        user_input=question, response=answer, retrieved_contexts=contexts
    )
    context_precision = await judge.context_precision.ascore(
        user_input=question, reference=reference, retrieved_contexts=contexts
    )
    context_recall = await judge.context_recall.ascore(
        user_input=question, retrieved_contexts=contexts, reference=reference
    )
    answer_relevancy = await judge.answer_relevancy.ascore(user_input=question, response=answer)
    return GenerationCaseMetrics(
        case_id=case_id,
        faithfulness=faithfulness.value,
        context_precision=context_precision.value,
        context_recall=context_recall.value,
        answer_relevancy=answer_relevancy.value,
    )


class GenerationSummary(BaseModel):
    """Mean RAGAS scores across every scored (answerable, non-refused) case."""

    faithfulness: float
    context_precision: float
    context_recall: float
    answer_relevancy: float
    case_count: int


def summarize_generation(per_case: list[GenerationCaseMetrics]) -> GenerationSummary:
    """Mean each RAGAS metric across `per_case`; all zero when nothing was scored."""
    n = len(per_case)
    if n == 0:
        return GenerationSummary(
            faithfulness=0.0,
            context_precision=0.0,
            context_recall=0.0,
            answer_relevancy=0.0,
            case_count=0,
        )
    return GenerationSummary(
        faithfulness=sum(c.faithfulness for c in per_case) / n,
        context_precision=sum(c.context_precision for c in per_case) / n,
        context_recall=sum(c.context_recall for c in per_case) / n,
        answer_relevancy=sum(c.answer_relevancy for c in per_case) / n,
        case_count=n,
    )


class PrecomputedScoreMetric(BaseMetric):
    """Wraps an already-computed RAGAS score as a DeepEval `BaseMetric`, so it gets
    DeepEval's threshold / `is_successful()` semantics for free. This is the
    "CI/test-runner integration layer" role `docs/architecture.md` §2.10 assigns to
    DeepEval, applied to a score RAGAS already computed rather than a second
    LLM-judge call for the same metric.
    """

    def __init__(self, name: str, score: float, threshold: float) -> None:
        self.name = name
        self.score = score
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        self.is_successful()
        assert self.score is not None
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        return self.measure(test_case)

    @property
    def __name__(self) -> str:
        return self.name
