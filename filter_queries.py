import json
import os

def filter_queries_by_results(tsv_file, in_queries_file, out_queries_file):
    """
    Reads QIDs from the results TSV and filters a queries JSONL file
    to only include those specific QIDs.
    """
    print(f"Reading valid QIDs from {tsv_file}...")
    valid_qids = set()
    try:
        with open(tsv_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    valid_qids.add(parts[0])
    except FileNotFoundError:
        print(f"Error: Could not find {tsv_file}")
        return

    print(f"Found {len(valid_qids)} unique QIDs in TSV.")

    print(f"Filtering {in_queries_file}...")
    matched_count = 0

    try:
        with open(in_queries_file, 'r', encoding='utf-8') as f_in, \
             open(out_queries_file, 'w', encoding='utf-8') as f_out:

            for line in f_in:
                data = json.loads(line)
                qid = str(data.get('_id', ''))

                if qid in valid_qids:
                    f_out.write(line)
                    matched_count += 1

        print(f"Success! Saved {matched_count} overlap queries to: {out_queries_file}")
    except Exception as e:
        print(f"Error processing JSONL: {e}")

if __name__ == "__main__":
    # You can update these paths if applying to other datasets later
    TSV_PATH = "../../../datasets/results/tree_scifact_b64_r128_k100/go_results.tsv"
    IN_QUERIES = "../../../datasets/scifact/queries.jsonl"
    OUT_QUERIES = "../../../datasets/scifact/queries_filtered.jsonl"

    filter_queries_by_results(TSV_PATH, IN_QUERIES, OUT_QUERIES)
