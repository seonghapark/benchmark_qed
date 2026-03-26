# Benchmark QED RAG (Local-Only)

This guide explains how to run `/storage/benchmark_qed/benchmark_qed_rag.py` fully locally.

The script retrieves from an already-built vector DB and can optionally generate answers with a local Hugging Face LLM.

## Required vs Optional

Required:

- `--output`: output JSON path.
- Reranker usage is mandatory in current code. A CrossEncoder reranker is always loaded and applied.

Optional (defaults exist):

- Retrieval/data: `--db-dir`, `--query`, `--selected-questions-file`, `--selected-questions-json-file`, `--queries-file`, `--limit`.
- Models: `--embedding-model`, `--reranker-model`, `--fisher-model`, `--llm-model`.
- Retrieval tuning: `--dense-top-k`, `--sparse-top-k`, `--final-top-k`, `--nprobe`.
- Hybrid/Fisher weighting: `--dense-weight`, `--sparse-weight`, `--semantic-weight`, `--fisher-weight`.
- Fisher config: `--fisher-modes` (or deprecated `--fisher-mode`), `--fisher-top-k`.
- Generation: `--llm-model`, `--max-new-tokens` (answer generation is always performed).
- Runtime/cache: `--cache-dir`.

## What This Script Uses

- Dense index: `/storage/hybrid_test_100_newdata/dense_ivfpq.faiss`
- Sparse index: `/storage/hybrid_test_100_newdata/sparse_bm25.pkl`
- Embedding model (default): `Qwen/Qwen3-Embedding-0.6B`
- Reranker model (default): `BAAI/bge-reranker-base` (always used)
- Optional local generator model (default): `Qwen/Qwen2.5-1.5B-Instruct`

No API keys are required.

## Query Input Sources

The script accepts questions from the following sources (in this order):

1. `--query` (single question)
2. `--selected-questions-file` (AutoQ selected text JSON)
3. `--selected-questions-json-file` (AutoQ selected objects JSON; extracts `text`)
4. `--queries-file` (`.jsonl`, `.json`, or text file)
4. Default fallback:
  - `/storage/qed/output/data_local_questions/selected_questions.json` if it exists
  - otherwise `/storage/qed/output/data_local_questions/selected_questions_text.json` if it exists
   - otherwise `/storage/questions.jsonl`

Duplicate questions from mixed sources are removed automatically.

## Basic Usage

Run retrieval only (single query):

```bash
python3 /storage/benchmark_qed/benchmark_qed_rag.py \
  --query "What are climate change impacts on glaciers?" \
  --output /storage/qed/output/rag_results_local.json
```

Run retrieval using AutoQ selected questions directly:

```bash
python3 /storage/benchmark_qed/benchmark_qed_rag.py \
  --selected-questions-file /storage/qed/output/data_local_questions/selected_questions_text.json \
  --limit 10 \
  --output /storage/qed/output/rag_results_selected_questions.json
```

Run retrieval using AutoQ selected question objects (extracts `text`):

```bash
python3 /storage/benchmark_qed/benchmark_qed_rag.py \
  --selected-questions-json-file /storage/qed/output/data_local_questions/selected_questions.json \
  --limit 10 \
  --output /storage/qed/output/rag_results_selected_questions_objects.json
```

Run retrieval + local answer generation:

```bash
python3 /storage/benchmark_qed/benchmark_qed_rag.py \
  --selected-questions-file /storage/benchmark_qed/output/data_local_questions/selected_questions_text.json \
  --limit 200 \
  --llm-model Qwen/Qwen2.5-1.5B-Instruct \
  --max-new-tokens 256 \
  --output /storage/benchmark_qed/output/rag_results_selected_questions_with_answers.json
```

Run retrieval with explicit final decision method (default is `dense_sparse_50_50`):

```bash
python3 /storage/benchmark_qed/benchmark_qed_rag.py \
  --selected-questions-file /storage/benchmark_qed/output/data_local_questions/selected_questions_text.json \
  --limit 20 \
  --final-retrieval-methods dense_100 \
  --output /storage/benchmark_qed/output/rag_results_dense100.json
```

Run retrieval with multiple final decision methods:

```bash
python3 /storage/benchmark_qed/benchmark_qed_rag.py \
  --selected-questions-file /storage/benchmark_qed/output/data_local_questions/selected_questions_text.json \
  --limit 20 \
  --final-retrieval-methods dense_100 sparse_100 dense_sparse_50_50 \
  --output /storage/benchmark_qed/output/rag_results_multi_method.json
```

Run retrieval using weighted dense+sparse hybrid as final decision:

```bash
python3 /storage/benchmark_qed/benchmark_qed_rag.py \
  --selected-questions-file /storage/benchmark_qed/output/data_local_questions/selected_questions_text.json \
  --limit 20 \
  --final-retrieval-methods dense_sparse_weighted \
  --dense-weight 0.7 \
  --sparse-weight 0.3 \
  --output /storage/benchmark_qed/output/rag_results_weighted_hybrid.json
```

Run retrieval using weighted semantic+fisher final decision:

```bash
python3 /storage/benchmark_qed/benchmark_qed_rag.py \
  --selected-questions-file /storage/benchmark_qed/output/data_local_questions/selected_questions_text.json \
  --limit 20 \
  --final-retrieval-methods semantic_fisher_weighted \
  --semantic-weight 0.6 \
  --fisher-weight 0.4 \
  --output /storage/benchmark_qed/output/rag_results_semantic_fisher_weighted.json
```

## Important Arguments

- `--db-dir`: Vector DB folder
- `--embedding-model`: Query embedding model (default `Qwen/Qwen3-Embedding-0.6B`)
- `--reranker-model`: CrossEncoder reranker model (default `BAAI/bge-reranker-base`).
- `--dense-top-k`: Initial dense candidates (default `20`)
- `--sparse-top-k`: Initial BM25 candidates (default `20`)
- `--final-top-k`: Final fused results per query (default `5`)
- `--dense-weight`: Fusion weight for dense score (default `0.5`)
- `--sparse-weight`: Fusion weight for sparse score (default `0.5`)
- `--final-retrieval-methods`: One or more final retrieval decision tracks. Choices:
  - `dense_sparse_50_50`
  - `dense_sparse_weighted`
  - `dense_100`
  - `sparse_100`
  - `semantic_100`
  - `fisher_100`
  - `semantic_fisher_50_50`
  - `semantic_fisher_weighted`
- `--semantic-weight`: Semantic weight used by `semantic_fisher_weighted` (default `0.5`)
- `--fisher-weight`: Fisher weight used by `semantic_fisher_weighted` (default `0.5`)
- `--fisher-modes`: One or more Fisher computation modes in one run. Choices: `none`, `embedding`, `entropy`, `gradient`.
- `--fisher-mode`: Deprecated single Fisher mode (still accepted for backward compatibility).
- `--fisher-model`: Local model used when `--fisher-mode` is `entropy` or `gradient`.
- `--fisher-top-k`: Candidate pool size used for expensive Fisher modes (`entropy`/`gradient`).
- `--nprobe`: FAISS IVF probing depth (default `32`)
- `--cache-dir`: Hugging Face cache location
- `--generate`: **Removed** — answer generation is now always performed.
- `--output`: Output JSON file path

Notes on final retrieval methods:

- `dense_sparse_weighted` uses `--dense-weight` and `--sparse-weight` for the final ranking.
- `semantic_fisher_weighted` uses `--semantic-weight` and `--fisher-weight` for the final ranking.
- `fisher_100`, `semantic_fisher_50_50`, and `semantic_fisher_weighted` require Fisher scoring to be enabled (`--fisher-modes` includes at least one of `embedding`, `entropy`, `gradient`).
- Recommended defaults for stable speed are `--fisher-modes embedding` or `--fisher-modes entropy` with a small `--fisher-top-k`.
- CrossEncoder reranker is always applied to candidate ordering in the current script.

## Output Format

The script writes a JSON list where each item includes:

- `question`
- `retrieved`: list of retrieved chunks with metadata and scores
- `selected_retrievals`: retrieval results for all selected final methods
- `final_retrieval_method`: method used to select `retrieved`
- `final_retrieval_methods`: all selected methods for this run
- `final_hybrid_weights`: dense/sparse weights used by weighted hybrid mode
- `retrieval_tracks`: all available top-k tracks for comparison
- `fisher_available`: whether Fisher-based tracks are usable for this run
- `fisher_key`: reserved compatibility field (currently `null`)
- `answer`: generated answer (empty when `--generate` is not used)

## Notes

- First run may be slower due to local model download/warmup.
- Retrieval-only mode is much faster than retrieval + generation.
- Keep `--limit` small for quick debugging.
