echo $SECONDS

cd /storage/araia/vector-search-engine/benchmark_qed/autoq_test && rm -rf output && PYTHONPATH="$PYTHONPATH:." benchmark-qed autoq settings.yaml output

echo $SECONDS
