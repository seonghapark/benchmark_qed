# AutoE Local Usage

This guide explains how to run `/storage/benchmark_qed/autoe.py` locally and what it produces.

`autoe.py` builds AutoE-style question-answer pairs from:

- an existing dense + BM25 index
- a directory of source JSON documents
- a directory of selected or candidate questions

It does not use a causal LLM to generate answers. Instead, it retrieves relevant chunks, optionally reranks them with a CrossEncoder, selects evidence sentences, and formats an extractive answer.

## What The Script Uses

- Dataset directory default: `/storage/data/new_samples`
- Question directory default: `/storage/qed/data_local_questions`
- Preferred index directory default: `/storage/hybrid_100_newdata`
- Fallback index directory default: `/storage/hybrid_test_100_newdata`
- Retrieval embedding model default: `Qwen/Qwen3-Embedding-0.6B`
- Answer ranking model default: `sentence-transformers/all-MiniLM-L6-v2`
- Reranker model default: `BAAI/bge-reranker-base`
- Output default: `/storage/qed/autoe_qa_pairs.json`

The script expects the index directory to contain both:

- `dense_ivfpq.faiss`
- `sparse_bm25.pkl`

## Question Inputs

`autoe.py` reads questions from the question directory in this order:

1. `selected_questions_text.json`
2. `selected_questions.json`
3. `candidate_questions.json`

Questions are deduplicated case-insensitively. If `--num-pairs` is larger than the number of available questions, the script raises an error.

## Dataset Expectations

The dataset directory should contain JSON files with at least a `title` field and useful text fields such as:

- `abstract`
- `text`
- `Introduction`
- `Methodology`
- `discussion`
- `conclusion`

The script builds a title-to-text map from those fields and uses it to expand evidence during answer construction.

## Basic Usage

Run with defaults:

```bash
python3 /storage/benchmark_qed/autoe.py
```

Run on 20 questions and save to a custom file:

```bash
python3 /storage/benchmark_qed/autoe.py \
  --num-pairs 20 \
  --output /storage/benchmark_qed/output/autoe_qa_pairs_20.json
```

Run with explicit dataset, question, and index directories:

```bash
python3 /storage/benchmark_qed/autoe.py \
  --dataset-dir /storage/data/new_samples \
  --question-dir /storage/qed/data_local_questions \
  --index-dir /storage/hybrid_100_newdata \
  --fallback-index-dir /storage/hybrid_test_100_newdata \
  --output /storage/benchmark_qed/output/autoe_qa_pairs.json
```

Run without CrossEncoder reranking:

```bash
python3 /storage/benchmark_qed/autoe.py \
  --no-reranker \
  --output /storage/benchmark_qed/output/autoe_no_reranker.json
```

Run with tuned retrieval settings:

```bash
python3 /storage/benchmark_qed/autoe.py \
  --num-pairs 10 \
  --dense-top-k 60 \
  --sparse-top-k 60 \
  --final-top-k 10 \
  --dense-weight 0.7 \
  --sparse-weight 0.3 \
  --nprobe 48 \
  --output /storage/benchmark_qed/output/autoe_tuned.json
```

## Important Arguments

- `--dataset-dir`: Directory of source JSON documents.
- `--question-dir`: Directory containing selected or candidate question files.
- `--index-dir`: Preferred vector index directory.
- `--fallback-index-dir`: Used if `--index-dir` does not contain both required index files.
- `--num-pairs`: Number of QA pairs to generate.
- `--embedding-model`: SentenceTransformer model used for query retrieval embeddings.
- `--answer-model`: SentenceTransformer model used to rank evidence sentences.
- `--reranker-model`: CrossEncoder model used for chunk reranking.
- `--no-reranker`: Disable CrossEncoder reranking.
- `--dense-top-k`: Number of dense candidates to fetch before fusion.
- `--sparse-top-k`: Number of BM25 candidates to fetch before fusion.
- `--final-top-k`: Number of fused chunks kept after ranking.
- `--nprobe`: FAISS IVF probe depth.
- `--dense-weight`: Weight of dense retrieval in hybrid scoring.
- `--sparse-weight`: Weight of sparse retrieval in hybrid scoring.
- `--output`: Output JSON path.

## How Answer Generation Works

For each question, the script:

1. Encodes the question with the retrieval embedding model.
2. Retrieves dense FAISS candidates.
3. Retrieves sparse BM25 candidates.
4. Normalizes and combines both score sets into a hybrid score.
5. Optionally reranks the fused chunks with a CrossEncoder.
6. Groups evidence by source title.
7. Splits document text and chunk text into sentences.
8. Ranks sentences with the answer model.
9. Selects diverse high-quality evidence sentences.
10. Formats the final answer as a short bullet-point summary.

This means answers are extractive and evidence-driven, not free-form LLM generations.

## Output Format

The script writes a JSON object with:

- `config`: run configuration summary
- `qa_pairs`: list of generated QA items

Each item in `qa_pairs` contains:

- `id`
- `question`
- `answer`
- `source_titles`
- `retrieved_chunks`

Each retrieved chunk contains:

- `chunk_idx`
- `text`
- `metadata`
- `scores`

The `scores` object can include:

- `dense_raw`
- `sparse_raw`
- `hybrid`

## Example Output Shape

```json
{
  "config": {
    "dataset_dir": "/storage/data/new_samples",
    "question_dir": "/storage/qed/data_local_questions",
    "index_dir": "/storage/hybrid_test_100_newdata",
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "answer_model": "sentence-transformers/all-MiniLM-L6-v2",
    "num_pairs": 10
  },
  "qa_pairs": [
    {
      "id": 1,
      "question": "...",
      "answer": "The retrieved documents highlight several findings related to the question.",
      "source_titles": ["..."],
      "retrieved_chunks": [
        {
          "chunk_idx": 123,
          "text": "...",
          "metadata": {
            "title": "..."
          },
          "scores": {
            "dense_raw": 0.82,
            "sparse_raw": 14.1,
            "hybrid": 0.73
          }
        }
      ]
    }
  ]
}
```

## Notes

- First run can be slower because models may need to be loaded or downloaded.
- The script runs entirely on CPU in its current implementation.
- If `--index-dir` is missing the two required index files, the fallback directory is used automatically.
- If no valid question files are found in the question directory, the script exits with a `FileNotFoundError`.
- If the dataset JSON files do not have usable `title` values, answer quality will degrade because evidence grouping depends on titles.