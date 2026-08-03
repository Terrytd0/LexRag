"""Reproducible retrieval experiment run backing `docs/experiments/retrieval_debugging.md`.

Indexes a small, hand-written set of representative contract-clause chunks
directly into Qdrant + Elasticsearch (bypassing PDF loading/chunking, which
already has its own Day 2 coverage -- this script exercises the Day 3
retrieval layer specifically), then runs a fixed set of sample queries
through dense-only, sparse-only, and hybrid retrieval, printing scores,
overlap, and latency for each.

Per `data/raw/sample_contracts/README.md`, synthetic documents must never be
added to that corpus directory -- this script keeps its fixture text
in-process instead of writing files there.

Usage (with `docker compose up -d mongo qdrant elasticsearch`):
    uv run python scripts/retrieval_debug_run.py
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from configs.logging import configure_logging
from configs.settings import get_settings
from domain.chunk import Chunk
from retrieval.dense import DenseRetriever
from retrieval.embedding import get_embedding_service
from retrieval.hybrid import HybridRetriever
from retrieval.keyword_store.elasticsearch_store import (
    ElasticsearchKeywordStore,
    get_elasticsearch_client,
)
from retrieval.sparse import SparseRetriever
from retrieval.vector_store.qdrant_store import QdrantVectorStore, get_qdrant_client

logger = logging.getLogger(__name__)

_CLAUSES: list[tuple[str, str, str]] = [
    # (doc_id, section, text)
    (
        "msa-vendorco",
        "Indemnification",
        "Each party shall indemnify, defend, and hold harmless the other party "
        "from and against any third-party claims, losses, or damages arising "
        "out of the indemnifying party's gross negligence or willful misconduct "
        "in the performance of this Agreement.",
    ),
    (
        "msa-vendorco",
        "Termination",
        "Either party may terminate this Agreement for convenience upon sixty "
        "(60) days' prior written notice to the other party, without cause and "
        "without penalty, subject to Section 8.3 (Wind-Down Obligations).",
    ),
    (
        "msa-vendorco",
        "Confidentiality",
        "Each party agrees to maintain the confidentiality of the other party's "
        "Confidential Information and not to disclose it to any third party "
        "without prior written consent, except as required by law.",
    ),
    (
        "msa-vendorco",
        "Governing Law",
        "This Agreement shall be governed by and construed in accordance with "
        "the laws of the State of Delaware, without regard to its conflict of "
        "laws principles.",
    ),
    (
        "nda-partnerco",
        "Definition",
        '"Confidential Information" means any non-public technical, business, '
        "or financial information disclosed by either party, whether oral, "
        "written, or in electronic form, that is designated as confidential.",
    ),
    (
        "nda-partnerco",
        "Term",
        "The obligations of confidentiality under this Agreement shall survive "
        "termination of this Agreement for a period of five (5) years from the "
        "date of disclosure.",
    ),
    (
        "employment-agreement",
        "Non-Compete",
        "During the term of employment and for twelve (12) months thereafter, "
        "the Employee shall not, directly or indirectly, engage in any business "
        "that competes with the Company within the Restricted Territory.",
    ),
    (
        "employment-agreement",
        "Compensation",
        "The Company shall pay the Employee a base salary payable in "
        "accordance with the Company's standard payroll practices, subject to "
        "applicable withholding.",
    ),
    (
        "saas-agreement",
        "Limitation of Liability",
        "In no event shall either party's aggregate liability arising out of "
        "or related to this Agreement exceed the total fees paid by Customer "
        "in the twelve (12) months preceding the claim, except for breaches of "
        "Section 8.3 (Confidentiality).",
    ),
    (
        "saas-agreement",
        "Force Majeure",
        "Neither party shall be liable for any failure or delay in performance "
        "under this Agreement due to causes beyond its reasonable control, "
        "including acts of God, war, or governmental action.",
    ),
    (
        "saas-agreement",
        "Payment Terms",
        "Customer shall pay all undisputed invoiced amounts within thirty (30) "
        "days of the invoice date. Late payments accrue interest at 1.5% per "
        "month or the maximum rate permitted by law.",
    ),
    (
        "licensing-agreement",
        "IP Assignment",
        "All intellectual property rights in any work product created by "
        "Contractor under this Agreement shall be assigned to and become the "
        "sole property of the Company upon creation.",
    ),
    (
        "licensing-agreement",
        "Dispute Resolution",
        "Any dispute arising out of or relating to this Agreement shall be "
        "resolved through binding arbitration administered by the American "
        "Arbitration Association under its Commercial Arbitration Rules.",
    ),
]

_QUERIES = [
    "How can either party end the agreement early without cause?",
    "Who is responsible for losses if something goes wrong during the project?",
    "What happens if an employee starts working for a competitor?",
    "Section 8.3",
    "What is the interest rate on late invoice payments?",
    "How are disagreements between the parties resolved?",
]


@dataclass
class _QueryOutcome:
    query: str
    dense_ids: list[str]
    sparse_ids: list[str]
    hybrid_ids: list[str]
    dense_latency: float
    sparse_latency: float
    hybrid_latency: float


def _build_chunks() -> list[Chunk]:
    chunks = []
    for index, (doc_id, section, text) in enumerate(_CLAUSES):
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:{index}",
                doc_id=doc_id,
                chunk_index=index,
                text=text,
                token_count=len(text.split()),
                page_number=1,
                section=section,
                source_filename=f"{doc_id}.pdf",
            )
        )
    return chunks


def _timed_dense(retriever: DenseRetriever, query: str, top_k: int) -> tuple[list[str], float]:
    start = time.monotonic()
    results = retriever.retrieve(query, top_k=top_k)
    return [r.chunk.chunk_id for r in results], time.monotonic() - start


def _timed_sparse(retriever: SparseRetriever, query: str, top_k: int) -> tuple[list[str], float]:
    start = time.monotonic()
    results = retriever.retrieve(query, top_k=top_k)
    return [r.chunk.chunk_id for r in results], time.monotonic() - start


async def _timed_hybrid(
    retriever: HybridRetriever, query: str, top_k: int
) -> tuple[list[str], float]:
    start = time.monotonic()
    results = await retriever.retrieve(query, top_k=top_k)
    return [r.chunk.chunk_id for r in results], time.monotonic() - start


async def main() -> None:
    configure_logging()
    settings = get_settings()
    top_k = 5

    embedding_service = get_embedding_service()
    vector_store = QdrantVectorStore(get_qdrant_client(), embedding_service, settings)
    keyword_store = ElasticsearchKeywordStore(get_elasticsearch_client(), settings)

    chunks = _build_chunks()
    print(f"Indexing {len(chunks)} sample clauses into Qdrant + Elasticsearch...")
    index_start = time.monotonic()
    vector_store.index_chunks(chunks)
    keyword_store.index_chunks(chunks)
    print(f"Indexing complete in {time.monotonic() - index_start:.2f}s\n")

    dense = DenseRetriever(vector_store, embedding_service, settings)
    sparse = SparseRetriever(keyword_store, settings)
    hybrid = HybridRetriever(dense, sparse, settings)

    outcomes: list[_QueryOutcome] = []
    for query in _QUERIES:
        dense_ids, dense_latency = _timed_dense(dense, query, top_k)
        sparse_ids, sparse_latency = _timed_sparse(sparse, query, top_k)
        hybrid_ids, hybrid_latency = await _timed_hybrid(hybrid, query, top_k)
        outcomes.append(
            _QueryOutcome(
                query,
                dense_ids,
                sparse_ids,
                hybrid_ids,
                dense_latency,
                sparse_latency,
                hybrid_latency,
            )
        )

        overlap = set(dense_ids) & set(sparse_ids)
        print(f"Query: {query!r}")
        print(f"  dense  ({dense_latency * 1000:6.1f}ms): {dense_ids}")
        print(f"  sparse ({sparse_latency * 1000:6.1f}ms): {sparse_ids}")
        print(f"  hybrid ({hybrid_latency * 1000:6.1f}ms): {hybrid_ids}")
        print(f"  overlap: {len(overlap)}/{top_k} chunk ids in common between dense and sparse\n")

    avg_dense = sum(o.dense_latency for o in outcomes) / len(outcomes)
    avg_sparse = sum(o.sparse_latency for o in outcomes) / len(outcomes)
    avg_hybrid = sum(o.hybrid_latency for o in outcomes) / len(outcomes)
    print("--- Summary ---")
    print(f"avg dense latency:  {avg_dense * 1000:.1f}ms")
    print(f"avg sparse latency: {avg_sparse * 1000:.1f}ms")
    print(f"avg hybrid latency: {avg_hybrid * 1000:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
