#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/minimal_settings_fast.yaml"
OUTPUT_DIR="/storage/output_fast"

# Use explicit binary path fallback if benchmark-qed is not on PATH.
if command -v benchmark-qed >/dev/null 2>&1; then
  BENCHMARK_QED_BIN="$(command -v benchmark-qed)"
elif [[ -x "/root/.local/bin/benchmark-qed" ]]; then
  BENCHMARK_QED_BIN="/root/.local/bin/benchmark-qed"
else
  echo "benchmark-qed not found. Install with: python3 -m pip install --user benchmark-qed" >&2
  exit 127
fi

# Force fresh sampling so updated sampling config can generate more questions.
rm -f "${OUTPUT_DIR}/sample_texts.parquet" \
      "${OUTPUT_DIR}/text_units.parquet" \
      "${OUTPUT_DIR}/documents.parquet"

mkdir -p "${OUTPUT_DIR}"

cd "${SCRIPT_DIR}"
OPENBLAS_NUM_THREADS=32 \
OMP_NUM_THREADS=32 \
MKL_NUM_THREADS=32 \
PYTHONPATH="${SCRIPT_DIR}" \
"${BENCHMARK_QED_BIN}" autoq \
  "${CONFIG_PATH}" \
  "${OUTPUT_DIR}" \
  --generation-types data_local


# 1. Using _normalized_input
# - Advantage: compatibility and stability.
# - Why: benchmark-qed expects each .json file to be a single object; your raw corpus can include list-shaped or inconsistent files.
# - Result: fewer parser/runtime failures, predictable ingestion, cleaner sampling coverage.

# 2. Smaller chunk_size
# - Advantage: more focused chunks.
# - Why: shorter chunks usually isolate one idea/fact, so retrieval and question grounding can be tighter.
# - Tradeoff: too small can lose context and increase fragment noise; usually improves specificity but can hurt completeness.

# 3. Bigger num_clusters
# - Advantage: broader topical diversity.
# - Why: clustering partitions the corpus into more groups, so sampling can touch more semantic regions.
# - Result: generated questions are less repetitive and cover more subtopics.
# - Tradeoff: more compute/time.

# 4. Bigger num_samples_per_cluster
# - Advantage: deeper coverage within each topic.
# - Why: once a cluster is selected, more samples from it expose more document variants and details.
# - Result: richer question pool and better chance of reaching high requested question counts.
# - Tradeoff: more compute/time and potentially more near-duplicate candidates if too high.

# Practical intuition:
# - num_clusters controls breadth.
# - num_samples_per_cluster controls depth.
# - chunk_size controls granularity.
# - _normalized_input controls data correctness for the pipeline.