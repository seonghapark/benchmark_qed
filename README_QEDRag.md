# Benchmark QED RAG (Local-Only)

This guide explains how to run `/storage/benchmark_qed/benchmark_qed_rag.py` fully locally.

The script retrieves from an already-built vector DB and can optionally generate answers with a local Hugging Face LLM.

## What This Script Uses

- Dense index: `/storage/hybrid_test_100_newdata/dense_ivfpq.faiss`
- Sparse index: `/storage/hybrid_test_100_newdata/sparse_bm25.pkl`
- Embedding model (default): `Qwen/Qwen3-Embedding-0.6B`
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
  --selected-questions-file /storage/qed/output/data_local_questions/selected_questions_text.json \
  --limit 10 \
  --generate \
  --llm-model Qwen/Qwen2.5-1.5B-Instruct \
  --max-new-tokens 128 \
  --output /storage/qed/output/rag_results_selected_questions_with_answers.json
```

## Important Arguments

- `--db-dir`: Vector DB folder
- `--embedding-model`: Query embedding model (default `Qwen/Qwen3-Embedding-0.6B`)
- `--dense-top-k`: Initial dense candidates (default `20`)
- `--sparse-top-k`: Initial BM25 candidates (default `20`)
- `--final-top-k`: Final fused results per query (default `5`)
- `--dense-weight`: Fusion weight for dense score (default `0.5`)
- `--sparse-weight`: Fusion weight for sparse score (default `0.5`)
- `--nprobe`: FAISS IVF probing depth (default `32`)
- `--cache-dir`: Hugging Face cache location
- `--generate`: Enable local text generation for answers
- `--output`: Output JSON file path

## Output Format

The script writes a JSON list where each item includes:

- `question`
- `retrieved`: list of retrieved chunks with metadata and scores
- `answer`: generated answer (empty when `--generate` is not used)

## Notes

- First run may be slower due to local model download/warmup.
- Retrieval-only mode is much faster than retrieval + generation.
- Keep `--limit` small for quick debugging.
