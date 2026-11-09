#!/bin/bash

# Update this path if your files are actually in datasets/results/results/
TARGET_DIR="../../../datasets/results" 

for dir in "$TARGET_DIR"/*/; do
    folder_name=$(basename "$dir")
    
    if [[ "$folder_name" == *bins* || "$folder_name" == *pacmann* ]]; then
        INPUT_JSON="${dir}results_reRank.json"
        OUTPUT_TSV="${dir}go_results.tsv"
        
        if [[ -f "$INPUT_JSON" ]]; then
            echo "Attempting to process: $INPUT_JSON"
            
            # Explicitly calling python3
            python3 json_to_tsv.py "$INPUT_JSON" "$OUTPUT_TSV"
            
        else
            echo "WARNING: No results_reRank.json found inside $folder_name"
        fi
    fi
done

echo "Batch conversion loop finished!"
