# Retrieval evaluation

The evaluation runner measures the current **hybrid retrieval** pipeline (semantic + BM25 + reciprocal-rank fusion) against manually annotated relevant chunk indexes. It reports Recall@K, MRR, and NDCG@K. It does not score answer faithfulness or claim quality improvements.

## Prepare a small gold set

1. Ensure the target PDFs are indexed in your local PostgreSQL database.
2. From `backend/`, activate the project virtual environment.
3. Inspect the chunk indexes and text for a document:

   ```bash
   PYTHONPATH=. python -m app.services.evaluation.run_evaluation --list-chunks RAG-Survey.pdf
   ```

4. Edit `backend/data/evaluation/retrieval_dataset.json`. Add question cases and the chunk indexes that genuinely contain evidence needed to answer each question:

   ```json
   {
     "id": "rag-001",
     "document_filename": "RAG-Survey.pdf",
     "question": "What is retrieval-augmented generation?",
     "relevant_chunk_indices": [2, 3]
   }
   ```

   The indexes above are illustrative only—verify them against your own indexed PDF before using them as labels. Include multiple relevant chunks when the answer spans passages.

5. Run the evaluation:

   ```bash
   PYTHONPATH=. python -m app.services.evaluation.run_evaluation
   ```

   Optional: use `-k 10` or `--dataset path/to/another.json`.

The runner resolves the document by filename, scopes retrieval to that document, and prints per-question results plus macro averages. Keep the same annotated questions and labels when comparing retrieval changes. The dataset is intentionally empty initially so no ground-truth labels are fabricated.
