
"""Compare retrieval stages on a human-annotated PDF question set.

From backend/:
    PYTHONPATH=. python -m app.services.evaluation.run_evaluation

Use --list-chunks FILENAME to inspect chunk indexes before annotating the dataset.
"""

import argparse
import json
from pathlib import Path
from statistics import mean

from app.db.connection import get_connection
from app.services.evaluation.metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.services.retrieval.bm25_search import bm25_search
from app.services.retrieval.hybrid_search import hybrid_search
from app.services.retrieval.reranker import Reranker
from app.services.retrieval.semantic_search import semantic_search


DATASET_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "evaluation"
    / "retrieval_dataset.json"
)

METHODS = ("semantic", "bm25", "hybrid", "reranked")


def _document_id(filename: str) -> int | None:
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM documents WHERE filename = %s",
                (filename,),
            )
            row = cursor.fetchone()
            return row[0] if row else None
    finally:
        connection.close()


def _list_chunks(filename: str) -> None:
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.chunk_index, c.page_number, c.text
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.filename = %s
                ORDER BY c.chunk_index
                """,
                (filename,),
            )
            rows = cursor.fetchall()
    finally:
        connection.close()

    if not rows:
        print(f"No indexed chunks found for {filename!r}.")
        return

    for chunk_index, page_number, text in rows:
        print(
            f"\n--- chunk_index={chunk_index}, page={page_number} ---\n"
            f"{text}\n"
        )


def _score(ranked: list[dict], relevant: set[int], k: int) -> dict:
    indices = [item["chunk_index"] for item in ranked[:k]]

    return {
        "recall_at_k": recall_at_k(indices, relevant, k),
        "mrr": reciprocal_rank(indices, relevant),
        "ndcg_at_k": ndcg_at_k(
            indices,
            {index: 1.0 for index in relevant},
            k,
        ),
        "retrieved_chunk_indices": indices,
    }


def _failure_analysis(
    relevant: set[int],
    candidate_indices: list[int],
    reranked_indices: list[int],
) -> dict:
    """Identify whether relevant chunks were missed during retrieval or reranking."""

    candidate_relevant = relevant.intersection(candidate_indices)
    retained_relevant = relevant.intersection(reranked_indices)

    dropped_relevant = candidate_relevant - retained_relevant
    missing_from_candidates = relevant - candidate_relevant

    if missing_from_candidates:
        failure_type = "candidate_retrieval_failure"
    elif dropped_relevant:
        failure_type = "reranking_loss"
    else:
        failure_type = "all_relevant_chunks_retained"

    return {
        "candidate_recall_at_20": (
            len(candidate_relevant) / len(relevant)
        ),
        "candidate_chunk_indices": candidate_indices,
        "candidate_relevant_chunks": sorted(candidate_relevant),
        "reranker_retained": sorted(retained_relevant),
        "reranker_dropped": sorted(dropped_relevant),
        "missing_from_candidates": sorted(missing_from_candidates),
        "failure_type": failure_type,
    }


def _reranker_diagnostics(
    relevant: set[int],
    candidates: list[dict],
    reranked: list[dict],
    k: int,
) -> list[dict]:
    """Compare hybrid and reranker ranks for relevant chunks."""

    hybrid_ranks = {
        item["chunk_index"]: rank
        for rank, item in enumerate(candidates, start=1)
    }

    reranker_ranks = {
        item["chunk_index"]: rank
        for rank, item in enumerate(reranked, start=1)
    }

    reranker_scores = {
        item["chunk_index"]: item.get("reranker_score")
        for item in reranked
    }

    final_indices = {
        item["chunk_index"]
        for item in reranked[:k]
    }

    return [
        {
            "chunk_index": chunk_index,
            "hybrid_rank": hybrid_ranks.get(chunk_index),
            "reranker_rank": reranker_ranks.get(chunk_index),
            "reranker_score": reranker_scores.get(chunk_index),
            "survived_top_k": chunk_index in final_indices,
        }
        for chunk_index in sorted(relevant)
    ]

def _macro_metrics(rows: list[dict]) -> dict:
    return {
        "mean_recall_at_k": mean(
            row["recall_at_k"] for row in rows
        ),
        "mean_mrr": mean(
            row["mrr"] for row in rows
        ),
        "mean_ndcg_at_k": mean(
            row["ndcg_at_k"] for row in rows
        ),
    }


def run_evaluation(
    dataset_path: Path = DATASET_PATH,
    k: int = 5,
) -> dict:
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    dataset = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )

    cases = dataset.get("cases", [])

    if not cases:
        raise ValueError(
            f"No evaluation cases in {dataset_path}. "
            "Add questions and relevant_chunk_indices after inspecting "
            "your indexed PDF chunks with --list-chunks."
        )

    reranker = Reranker()

    per_method = {method: [] for method in METHODS}
    case_results = []

    for case in cases:
        filename = case["document_filename"]
        relevant = set(case["relevant_chunk_indices"])

        if not relevant:
            raise ValueError(
                f"Case {case.get('id', '?')} has no relevant_chunk_indices."
            )

        document_id = _document_id(filename)

        if document_id is None:
            raise ValueError(
                f"Document {filename!r} is not indexed in PostgreSQL."
            )

        question = case["question"]

        semantic = semantic_search(
            question,
            top_k=k,
            document_id=document_id,
        )

        bm25 = bm25_search(
            question,
            top_k=k,
            document_id=document_id,
        )

        hybrid_candidates = hybrid_search(
            question,
            top_k=max(k, 20),
            retrieval_k=max(k, 20),
            document_id=document_id,
        )

        hybrid = hybrid_candidates[:k]

        reranked_all = reranker.rerank(
            question,
            hybrid_candidates,
            top_k=len(hybrid_candidates),
        )

        reranked = reranked_all[:k]

        ranked_by_method = {
            "semantic": semantic,
            "bm25": bm25,
            "hybrid": hybrid,
            "reranked": reranked,
        }

        methods_result = {}

        for method, ranked in ranked_by_method.items():
            scored = _score(ranked, relevant, k)
            per_method[method].append(scored)
            methods_result[method] = scored

        candidate_indices = [
            item["chunk_index"]
            for item in hybrid_candidates
        ]

        reranked_indices = [
            item["chunk_index"]
            for item in reranked
        ]

        failure_analysis = _failure_analysis(
            relevant,
            candidate_indices,
            reranked_indices,
        )
        reranker_diagnostics = _reranker_diagnostics(
            relevant,
            hybrid_candidates,
            reranked_all,
            k,
        )

        case_results.append({
            "id": case.get("id"),
            "question": question,
            "relevant_chunk_indices": sorted(relevant),
            **failure_analysis,
            "reranker_diagnostics": reranker_diagnostics,
            "methods": methods_result,
        })

    failure_counts = {
        failure_type: sum(
            row["failure_type"] == failure_type
            for row in case_results
        )
        for failure_type in (
            "candidate_retrieval_failure",
            "reranking_loss",
            "all_relevant_chunks_retained",
        )
    }

    summary = {
        "cases": len(case_results),
        "k": k,
        "methods": {
            method: _macro_metrics(rows)
            for method, rows in per_method.items()
        },
        "failure_analysis": {
            "failure_type_counts": failure_counts,
            "mean_candidate_recall_at_20": mean(
                row["candidate_recall_at_20"]
                for row in case_results
            ),
        },
        "results": case_results,
    }

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    return summary


def run_rrf_sweep(
    dataset_path: Path = DATASET_PATH,
    k: int = 5,
    rrf_values: list[int] | None = None,
) -> dict:
    """Compare RRF constants without changing the production retrieval default."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    values = rrf_values or [10, 30, 60, 100]

    if not values or any(value <= 0 for value in values):
        raise ValueError(
            "All RRF constants must be greater than zero."
        )

    dataset = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )

    cases = dataset.get("cases", [])

    if not cases:
        raise ValueError(
            f"No evaluation cases in {dataset_path}. "
            "Add questions and relevant_chunk_indices after inspecting "
            "your indexed PDF chunks with --list-chunks."
        )

    reranker = Reranker()

    summaries = {}
    details = {}

    for rrf_k in values:
        hybrid_rows = []
        reranked_rows = []
        case_rows = []

        for case in cases:
            filename = case["document_filename"]
            document_id = _document_id(filename)

            if document_id is None:
                raise ValueError(
                    f"Document {filename!r} is not indexed in PostgreSQL."
                )

            relevant = set(case["relevant_chunk_indices"])

            if not relevant:
                raise ValueError(
                    f"Case {case.get('id', '?')} has no relevant_chunk_indices."
                )

            candidates = hybrid_search(
                case["question"],
                top_k=max(k, 20),
                retrieval_k=max(k, 20),
                rrf_k=rrf_k,
                document_id=document_id,
            )

            hybrid = _score(
                candidates[:k],
                relevant,
                k,
            )

            reranked_results = reranker.rerank(
                case["question"],
                candidates,
                top_k=k,
            )

            reranked = _score(
                reranked_results,
                relevant,
                k,
            )

            hybrid_rows.append(hybrid)
            reranked_rows.append(reranked)

            case_rows.append({
                "id": case.get("id"),
                "relevant_chunk_indices": sorted(relevant),
                "hybrid": hybrid,
                "reranked": reranked,
            })

        summaries[str(rrf_k)] = {
            method: _macro_metrics(rows)
            for method, rows in (
                ("hybrid", hybrid_rows),
                ("reranked", reranked_rows),
            )
        }

        details[str(rrf_k)] = case_rows

    result = {
        "cases": len(cases),
        "k": k,
        "rrf_constants": values,
        "macro_metrics": summaries,
        "per_case": details,
    }

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare semantic, BM25, hybrid, and reranked retrieval."
        )
    )

    parser.add_argument(
        "--list-chunks",
        metavar="FILENAME",
        help="Print indexed chunk indexes/text for a PDF.",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to a JSON evaluation dataset.",
    )

    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Evaluation cutoff (default: 5).",
    )

    parser.add_argument(
        "--rrf-sweep",
        action="store_true",
        help="Compare RRF constants 10, 30, 60, and 100.",
    )

    parser.add_argument(
        "--rrf-values",
        type=int,
        nargs="+",
        default=[10, 30, 60, 100],
        help="RRF constants used with --rrf-sweep.",
    )

    args = parser.parse_args()

    if args.k <= 0:
        parser.error("-k must be greater than zero.")

    if args.list_chunks:
        _list_chunks(args.list_chunks)
    elif args.rrf_sweep:
        run_rrf_sweep(
            args.dataset,
            args.k,
            args.rrf_values,
        )
    else:
        run_evaluation(
            args.dataset,
            args.k,
        )


if __name__ == "__main__":
    main()