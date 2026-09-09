import numpy as np
from pyserini.index import LuceneIndexReader
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True, help='Path to Lucene index')
    parser.add_argument('--aligned', required=True, help='Path to stage3_doc_align.npy')
    parser.add_argument('--original_ids', required=True, help='Path to stage3_doc.npy.ids')
    args = parser.parse_args()

    # IDs from your logs
    # Query 0 Gold (Worked): Internal 2465873
    # Query 1 Gold (Failed): Internal 6754452
    ids_to_check = [
        (2465873, "Query 0 Gold (Should be non-zero)"),
        (6754452, "Query 1 Gold (Was zero in logs)")
    ]

    print(f"Loading aligned embeddings from {args.aligned}...")
    aligned = np.load(args.aligned, mmap_mode='r')
    reader = LuceneIndexReader(args.index)

    print(f"Loading original IDs from {args.original_ids}...")
    with open(args.original_ids) as f:
        # Read into a set for fast lookup. Strip whitespace to match alignment logic.
        original_ids = set(line.strip() for line in f)

    for internal_id, desc in ids_to_check:
        print(f"\n--- Checking {desc}: Internal ID {internal_id} ---")
        
        # 1. Check the vector in the file
        vec = aligned[internal_id]
        norm = np.linalg.norm(vec)
        print(f"  Vector Norm in File: {norm}")
        print(f"  First 5 values: {vec[:5]}")
        
        # 2. Check the ID mapping
        try:
            ext_id = reader.convert_internal_docid_to_collection_docid(internal_id)
            print(f"  Maps to External ID: '{ext_id}'")
            
            if ext_id in original_ids:
                print(f"  [OK] External ID found in original IDs file.")
            else:
                print(f"  [FAIL] External ID NOT FOUND in original IDs file! (This is why it's zero)")
        except Exception as e:
            print(f"  [ERROR] Could not convert ID: {e}")

if __name__ == "__main__":
    main()