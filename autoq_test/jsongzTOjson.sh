#!/bin/bash
# Convert json.gz files to json files

SRC_DIR="/data/araia/peS2o/data/v2_val"
DEST_DIR="/storage/tttmp/pes2o_v2val_json"

mkdir -p "$DEST_DIR"

for gz_file in "$SRC_DIR"/*.json.gz; do
    if [ -f "$gz_file" ]; then
        filename=$(basename "$gz_file" .gz)
        echo "Converting: $gz_file -> $DEST_DIR/$filename"
        zcat "$gz_file" > "$DEST_DIR/$filename"
    fi
done

echo "Done. Files in $DEST_DIR:"
ls -lh "$DEST_DIR"
