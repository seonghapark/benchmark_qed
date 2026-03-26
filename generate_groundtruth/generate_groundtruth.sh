set -e
cd /storage/araia/vector-search-engine/benchmark_qed/generate_groundtruth
PY=.env/bin/python
if [ ! -x "$PY" ]; then PY=/usr/bin/python; fi
for CASE in 1 2 3 4 5 6; do
  IDX="/storage/araia/araia_model2/case${CASE}"
  "$PY" groundtruth.py \
    --questions-glob "/storage/output_fast_runs/seed_4[3-6]/data_local_questions/selected_questions_text.json" \
    --dataset-dir /storage/araia/vector-search-engine/data/new_samples \
    --index-dir "$IDX" \
    --fallback-index-dir "$IDX" \
    --output-name "groundtruth_answers_case${CASE}.json" \
    --qa-only-name "groundtruth_qa_only_case${CASE}.json"

  for SEED in 43 44 45 46; do
    SRC_DIR="/storage/output_fast_runs/seed_${SEED}/data_local_questions"
    cp "$SRC_DIR/groundtruth_answers_case${CASE}.json" "/storage/araia/araia_model2/case${CASE}/seed_${SEED}_groundtruth_answers.json"
    cp "$SRC_DIR/groundtruth_qa_only_case${CASE}.json" "/storage/araia/araia_model2/case${CASE}/seed_${SEED}_groundtruth_qa_only.json"
  done
  echo "Finished case${CASE}"
done

cd /storage/araia/vector-search-engine/benchmark_qed/generate_groundtruth && PY=.env/bin/python && if [ ! -x "$PY" ]; then PY=/usr/bin/python; fi && echo "PY=$PY" && "$PY" groundtruth.py --questions-glob "/storage/output_fast_runs/seed_4[3-6]/data_local_questions/selected_questions_text.json" --dataset-dir /storage/araia/vector-search-engine/data/new_samples --index-dir /storage/araia/araia_model2/case1 --fallback-index-dir /storage/araia/araia_model2/case1 --output-name groundtruth_answers_case1.json --qa-only-name groundtruth_qa_only_case1.json

