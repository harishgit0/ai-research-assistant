# Retrieval evaluation

The evaluation runner compares four retrieval stages against manually annotated relevant chunk indexes:

- **Semantic:** embedding similarity search.
- **BM25:** lexical retrieval.
- **Hybrid:** semantic + BM25 combined with reciprocal-rank fusion (RRF).
- **Reranked:** hybrid candidates reordered by the cross-encoder reranker.

It reports Recall@K, MRR, and NDCG@K per method and per question. It evaluates retrieval ranking—not answer faithfulness, factual correctness, or claim quality.

## Run

Ensure the target PDF is indexed in your local PostgreSQL database. From `backend/`, activate the project virtual environment, then run:

```bash
PYTHONPATH=. python -m app.services.evaluation.run_evaluation
```

Use `-k 10` to change the evaluation cutoff, or `--dataset path/to/another.json` to select another dataset.

## Inspect and label chunks

```bash
PYTHONPATH=. python -m app.services.evaluation.run_evaluation --list-chunks RAG-Survey.pdf
```

Edit `backend/data/evaluation/retrieval_dataset.json`. Each case contains an ID, indexed document filename, question, and `relevant_chunk_indices`. Assign labels only after reading the chunk text; include multiple relevant chunks when evidence spans passages. Keep the same gold set when comparing pipeline changes.

## Interpretation

Recall@K measures whether labeled relevant chunks appear in the first K results. MRR rewards an earlier first relevant result; NDCG@K measures ranking quality. With a small manually labeled dataset, treat scores as diagnostic—not general performance guarantees. Expand the dataset with diverse documents and questions before drawing broad conclusions.

The runner loads the reranker for each run and retrieves hybrid candidates up to max(K, 20) before reranking to K results. Semantic, BM25, and hybrid baselines are each evaluated at K.
