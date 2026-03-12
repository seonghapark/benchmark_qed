# Generate Questions with Benchmark-QED (Local Setup)

This guide explains how to generate questions with `benchmark-qed` using the local Hugging Face provider configuration in this workspace.

## Prerequisites

- Python 3.11+
- `benchmark-qed` installed
- Local provider module available at `/storage/local_hf_provider.py`
- Config file at `/storage/settings.yaml`
- Input data folder at `/storage/data/new_samples_local`

Install Benchmark-QED if needed:

```bash
python3 -m pip install benchmark-qed
```

## 1. Verify Your Config

Current config file:

- `/storage/settings.yaml`

Important fields to check:

- `input.dataset_path` points to your JSON folder (currently `/storage/data/new_samples_local`)
- `chat_model.llm_provider` is `local.hf.chat`
- `embedding_model.llm_provider` is `local.hf.embedding`
- `custom_providers` entries reference:
  - `module: local_hf_provider`
  - `model_class: HuggingFaceLocalChat`
  - `model_class: HuggingFaceLocalEmbedding`

## 2. Run Data-Local Question Generation

Use `PYTHONPATH=/storage` so Benchmark-QED can import the local provider module:

```bash
cd /storage
PYTHONPATH=/storage benchmark-qed autoq /storage/settings.yaml /storage/output --generation-types data_local
```

## 3. Output Files

Generated files are written to:

- `/storage/qed/output/data_local_questions/candidate_questions.json`
- `/storage/qed/output/data_local_questions/selected_questions.json`
- `/storage/qed/output/data_local_questions/selected_questions_text.json`

Sampling artifacts are written to:

- `/storage/qed/output/documents.parquet`
- `/storage/qed/output/text_units.parquet`
- `/storage/qed/output/sample_texts.parquet`

## 4. Regenerate with New Sampling

If you change chunking/sampling settings and want a fresh sample, delete cached sample files first:

```bash
rm -f /storage/qed/output/sample_texts.parquet \
      /storage/qed/output/text_units.parquet \
      /storage/qed/output/documents.parquet
```

Then run the AutoQ command again.

## 5. Performance Tips (Local)

If generation is slow, reduce the workload in `/storage/settings.yaml`:

- Lower `sampling.num_clusters`
- Lower `sampling.num_samples_per_cluster`
- Lower `data_local.num_questions`
- Lower `chat_model.call_args.max_new_tokens`
- Keep `chat_model.concurrent_requests: 1` for stability on limited VRAM

## 6. Common Issues

- `ModuleNotFoundError: local_hf_provider`
  - Run with `PYTHONPATH=/storage`
- Slow first run
  - Model download and warm-up can take time
- Reusing old sample after config changes
  - Remove parquet cache files in `/storage/qed/output`
