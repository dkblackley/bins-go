from collections import defaultdict
import json
import glob
import os

def load_qrels(file_path):
    """Parses ground truth data, handling both 3-column and 4-column formats."""
    qrels = defaultdict(set)
    with open(file_path, 'r') as f:
        for line in f:
            # Skip header rows if they exist
            if line.startswith("query"):
                continue
                
            parts = line.strip().split()
            
            # Route based on MS MARCO (4 cols) vs SciFact/TREC-COVID (3 cols)
            if len(parts) == 4: 
                qid, _, docid, rel = parts
            elif len(parts) == 3:
                qid, docid, rel = parts
            else:
                continue

            # Assume any score > 0 indicates a relevant document
            if float(rel) > 0: 
                qrels[qid].add(docid)
                
    return qrels

def load_results(file_path):
    """Parses the results TSV file into a dictionary of ranked lists."""
    results = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                qid, docid = parts[0], parts[1]
                # Appends maintain the ranked order assuming the file is sorted by score
                results[qid].append(docid) 
    return results


def calculate_metrics(qrels, results):
    """Calculates MRR and Recall for the provided results against ground truth."""
    mrr_sum = 0.0
    recall_sum = 0.0
    num_queries = 0

    for qid, relevant_docs in qrels.items():
        # Only evaluate queries that actually have relevant documents
        if not relevant_docs:
            continue
            
        num_queries += 1
        retrieved_docs = results.get(qid, [])

        # 1. Calculate MRR
        rr = 0.0
        for rank, docid in enumerate(retrieved_docs, start=1):
            if docid in relevant_docs:
                rr = 1.0 / rank
                break  # Stop at the first relevant document
        mrr_sum += rr

        # 2. Calculate Recall
        # Intersection of relevant docs and retrieved docs divided by total relevant
        retrieved_relevant = len(relevant_docs.intersection(set(retrieved_docs)))
        recall_sum += retrieved_relevant / len(relevant_docs)

    mrr = mrr_sum / num_queries if num_queries > 0 else 0.0
    recall = recall_sum / num_queries if num_queries > 0 else 0.0

    return mrr, recall

def update_metadata_json(json_path, mrr, recall):
    """Opens the target JSON, appends the new metrics, and saves it."""
    try:
        with open(json_path, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist yet, create an empty dictionary
        metadata = {}

    # Add the new keys
    metadata['MRR'] = mrr
    metadata['Recall'] = recall

    # Write back to the file
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)


# 1. Map the datasets to their specific ground truth (qrel) paths
qrel_paths = {
    "msmarco": "../../../datasets/msmarco/qrels/qrels.dev.small.tsv",
    "scifact": "../../../datasets/scifact/qrels/train.tsv",
    "trec-covid": "../../../datasets/trec-covid/qrels/test.tsv"
}

# 2. Find all directories matching the tree pattern
results_directories = glob.glob("../../../datasets/results/results/tree_*")

for results_dir in results_directories:
    # Identify which dataset this directory belongs to
    dataset_name = None
    for name in qrel_paths.keys():
        if name in results_dir:
            dataset_name = name
            break
    
    # Skip if the directory name doesn't match our known datasets
    if not dataset_name:
        print(f"Skipping {results_dir}: Could not match to 'msmarco', 'scifact', or 'trec-covid'.")
        continue
        
    qrel_path = qrel_paths[dataset_name]
    
    # Construct the file paths for this specific iteration
    results_path = os.path.join(results_dir, "go_results.tsv")
    metadata_path = os.path.join(results_dir, "metadata.json")
    
    # Safely skip the folder if there are no results to process
    if not os.path.exists(results_path):
        print(f"Skipping {results_dir}: No go_results.tsv found.")
        continue

    # 3. Load the data
    qrels_data = load_qrels(qrel_path)
    results_data = load_results(results_path)

    # 4. Compute metrics
    mrr_score, recall_score = calculate_metrics(qrels_data, results_data)

    # 5. Save to JSON
    update_metadata_json(metadata_path, mrr_score, recall_score)

    print(f"Updated {metadata_path} | Dataset: {dataset_name} | MRR: {mrr_score:.4f} | Recall: {recall_score:.4f}")
