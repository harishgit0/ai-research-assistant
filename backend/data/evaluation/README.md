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

## Current measured results

The current manually labeled dataset contains **4 questions** from `RAG-Survey.pdf`, each with one labeled relevant chunk. At **K=5**, the recorded results are:

| Method | Mean Recall@5 | Mean MRR | Mean NDCG@5 |
|---|---:|---:|---:|
| Semantic | 0.750 | 0.5833 | 0.6250 |
| BM25 | 0.750 | 0.7500 | 0.7500 |
| Hybrid (RRF=60) | 0.750 | 0.6250 | 0.6577 |
| Reranked | 1.000 | 0.8750 | 0.9077 |

These are observed results on this small dataset, not general performance guarantees. In particular, a perfect mean Recall@5 here means each of these four labeled chunks appeared in its case's top five; it does not establish broad retrieval reliability.

## RRF parameter experiment

Run an evaluation-only sweep over RRF constants 10, 30, 60, and 100:

```bash
PYTHONPATH=. python -m app.services.evaluation.run_evaluation --rrf-sweep
```

To choose values explicitly, add `--rrf-values 10 30 60 100`. The sweep reports hybrid and reranked metrics for each value; it does not modify the production `hybrid_search` default (`rrf_k=60`).

### Observed sweep (K=5, 4 cases)

| RRF constant | Hybrid Recall@5 | Hybrid MRR | Hybrid NDCG@5 | Reranked MRR | Reranked NDCG@5 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.750 | 0.625 | 0.6577 | 0.875 | 0.9077 |
| 30 | 0.750 | 0.625 | 0.6577 | 0.875 | 0.9077 |
| 60 | 0.750 | 0.625 | 0.6577 | 0.875 | 0.9077 |
| 100 | 0.750 | 0.625 | 0.6577 | 0.875 | 0.9077 |

The top-five aggregate metrics were identical across tested constants. Some lower-ranked candidate ordering changed between settings, but this experiment provides no measured basis to replace the existing RRF=60 default. This is not evidence that the constants are equivalent on other datasets.

## Inspect and label chunks

```bash
PYTHONPATH=. python -m app.services.evaluation.run_evaluation --list-chunks RAG-Survey.pdf
```

Edit `backend/data/evaluation/retrieval_dataset.json`. Each case contains an ID, indexed document filename, question, and `relevant_chunk_indices`. Assign labels only after reading the chunk text; include multiple relevant chunks when evidence spans passages. Keep the same gold set when comparing pipeline changes.

## Interpretation and limitations

Recall@K measures whether labeled relevant chunks appear in the first K results. MRR rewards an earlier first relevant result; NDCG@K measures ranking quality. With one relevant chunk per case, these metrics are especially limited in what they say about multi-passage questions.

Expand the dataset with more questions, multiple relevant chunks where appropriate, and diverse documents before drawing broad conclusions. Compare on the same labeled cases and cutoff. The runner retrieves hybrid candidates up to `max(K, 20)` before reranking to K results; semantic, BM25, and hybrid baselines are each evaluated at K.