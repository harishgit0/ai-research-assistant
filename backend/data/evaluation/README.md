# Retrieval evaluation

The evaluation runner compares four retrieval stages against the chunk labels in retrieval_dataset.json:

- **Semantic:** embedding similarity search.
- **BM25:** lexical retrieval.
- **Hybrid:** semantic + BM25 combined with reciprocal-rank fusion (RRF).
- **Reranked:** hybrid candidates reordered by the cross-encoder reranker.

It reports Recall@K, MRR, and NDCG@K per method and per question. It evaluates retrieval ranking against the supplied labels, not answer faithfulness, factual correctness, or claim quality.

## Run

Ensure the target PDF is indexed in your configured PostgreSQL database. From backend/, activate the virtual environment and run:

    PYTHONPATH=. python -m app.services.evaluation.run_evaluation

Use -k 10 to change the cutoff, --dataset path/to/another.json to select a dataset, or --rrf-sweep to compare RRF constants.

## Dataset status

The current JSON dataset contains 35 question cases for RAG-Survey.pdf. Its relevant_chunk_indices are manual labels tied to the exact PDF extraction and chunking output. Before reporting benchmark metrics, validate every labeled index against the actual indexed chunks for the exact document version. Do not infer or change gold chunk IDs based only on question wording.

List the chunks from your local database before reviewing labels:

    PYTHONPATH=. python -m app.services.evaluation.run_evaluation --list-chunks RAG-Survey.pdf

Edit retrieval_dataset.json only after inspecting the matching chunk text. Add multiple relevant chunk indexes when a question requires evidence from more than one passage. Keep the same reviewed label set when comparing retrieval approaches.

## Metrics

- Recall@K measures the fraction of labeled relevant chunks retrieved in the first K results.
- MRR rewards a higher rank for the first relevant result.
- NDCG@K measures ranking quality using relevance labels.

The current runner uses binary relevant/not-relevant labels for these metrics. Results depend on the document, chunking settings, label quality, and dataset size. Do not treat scores from this dataset as general performance guarantees.

The runner retrieves up to max(K, 20) hybrid candidates before reranking. Semantic and BM25 baselines are evaluated at K; hybrid results are evaluated at K; reranked results are produced from the hybrid candidate pool.

## RRF parameter sweep

Run:

    PYTHONPATH=. python -m app.services.evaluation.run_evaluation --rrf-sweep

Or provide explicit constants:

    PYTHONPATH=. python -m app.services.evaluation.run_evaluation --rrf-sweep --rrf-values 10 30 60 100

The sweep reports hybrid and reranked metrics for each constant and does not change the production default of RRF=60.
