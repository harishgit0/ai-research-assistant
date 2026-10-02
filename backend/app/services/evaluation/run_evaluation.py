"""Compare retrieval stages on a human-annotated PDF question set.

From backend/: PYTHONPATH=. python -m app.services.evaluation.run_evaluation
Use --list-chunks FILENAME to inspect chunk indexes before annotating the dataset.
"""
import argparse
import json
from pathlib import Path
from statistics import mean

from app.db.connection import get_connection
from app.services.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from app.services.retrieval.bm25_search import bm25_search
from app.services.retrieval.hybrid_search import hybrid_search
from app.services.retrieval.reranker import Reranker
from app.services.retrieval.semantic_search import semantic_search

DATASET_PATH = Path(__file__).resolve().parents[3] / "data" / "evaluation" / "retrieval_dataset.json"
METHODS = ("semantic", "bm25", "hybrid", "reranked")


def _document_id(filename: str) -> int | None:
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id FROM documents WHERE filename = %s", (filename,))
            row = cursor.fetchone()
            return row[0] if row else None
    finally:
        connection.close()


def _list_chunks(filename: str) -> None:
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT c.chunk_index, c.page_number, c.text
                   FROM chunks c JOIN documents d ON d.id = c.document_id
                   WHERE d.filename = %s ORDER BY c.chunk_index""",
                (filename,),
            )
            rows = cursor.fetchall()
    finally:
        connection.close()
    if not rows:
        print(f"No indexed chunks found for {filename!r}.")
        return
    for chunk_index, page_number, text in rows:
        print(f"\\n--- chunk_index={chunk_index}, page={page_number} ---\\n{text}\\n")


def _score(ranked: list[dict], relevant: set[int], k: int) -> dict:
    indices = [item["chunk_index"] for item in ranked[:k]]
    return {
        "recall_at_k": recall_at_k(indices, relevant, k),
        "mrr": reciprocal_rank(indices, relevant),
        "ndcg_at_k": ndcg_at_k(indices, {index: 1.0 for index in relevant}, k),
        "retrieved_chunk_indices": indices,
    }


def run_evaluation(dataset_path: Path = DATASET_PATH, k: int = 5) -> dict:
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset.get("cases", [])
    if not cases:
        raise ValueError(
            f"No evaluation cases in {dataset_path}. Add questions and relevant_chunk_indices "
            "after inspecting your indexed PDF chunks with --list-chunks."
        )

    reranker = Reranker()
    per_method = {method: [] for method in METHODS}
    case_results = []

    for case in cases:
        filename = case["document_filename"]
        relevant = set(case["relevant_chunk_indices"])
        if not relevant:
            raise ValueError(f"Case {case.get('id', '?')} has no relevant_chunk_indices.")
        document_id = _document_id(filename)
        if document_id is None:
            raise ValueError(f"Document {filename!r} is not indexed in PostgreSQL.")

        question = case["question"]
        semantic = semantic_search(question, top_k=k, document_id=document_id)
        bm25 = bm25_search(question, top_k=k, document_id=document_id)
        hybrid_candidates = hybrid_search(
            question, top_k=max(k, 20), retrieval_k=max(k, 20), document_id=document_id
        )
        hybrid = hybrid_candidates[:k]
        reranked = reranker.rerank(question, hybrid_candidates, top_k=k)

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

        case_results.append({
            "id": case.get("id"),
            "question": question,
            "relevant_chunk_indices": sorted(relevant),
            "methods": methods_result,
        })

    summary = {
        "cases": len(case_results),
        "k": k,
        "methods": {
            method: {
                "mean_recall_at_k": mean(row["recall_at_k"] for row in rows),
                "mean_mrr": mean(row["mrr"] for row in rows),
                "mean_ndcg_at_k": mean(row["ndcg_at_k"] for row in rows),
            }
            for method, rows in per_method.items()
        },
        "results": case_results,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare semantic, BM25, hybrid, and reranked retrieval.")
    parser.add_argument("--list-chunks", metavar="FILENAME", help="Print indexed chunk indexes/text for a PDF.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Path to a JSON evaluation dataset.")
    parser.add_argument("-k", type=int, default=5, help="Evaluation cutoff (default: 5).")
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("-k must be greater than zero.")
    if args.list_chunks:
        _list_chunks(args.list_chunks)
    else:
        run_evaluation(args.dataset, args.k)


if __name__ == "__main__":
    main()
