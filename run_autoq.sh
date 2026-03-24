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
