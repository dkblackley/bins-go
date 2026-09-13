import argparse
import numpy as np
from pyserini.index import LuceneIndexReader
from tqdm import tqdm
import os

def main():
    parser = argparse.ArgumentParser(description="Align embeddings and IDs to Lucene Internal IDs")
    parser.add_argument('--index', required=True, help='Path to Lucene index')
    parser.add_argument('--embeddings', required=True, help='Path to original embeddings.npy')
    parser.add_argument('--ids', required=True, help='Path to original embeddings.npy.ids')
    parser.add_argument('--output', required=True, help='Path to output aligned_embeddings.npy')
    args = parser.parse_args()

    # 1. Load Mapping: External ID -> Original Row Index
    print(f"Loading ID mapping from {args.ids}...")
    ext_to_row = {}
    with open(args.ids, 'r') as f:
        for idx, line in enumerate(f):
            ext_to_row[line.strip()] = idx
    
    # 2. Open Original Embeddings (Read-only mmap)
    original = np.load(args.embeddings, mmap_mode='r')
    N, dim = original.shape
    print(f"Original embeddings shape: {original.shape}")

    # 3. Initialize Lucene Reader
    reader = LuceneIndexReader(args.index)
    num_docs = reader.stats()['documents']
    
    # 4. Prepare Outputs
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Aligned NPY
    aligned = np.lib.format.open_memmap(args.output, mode='w+', dtype=original.dtype, shape=(num_docs, dim))
    # Aligned IDs (matching the NPY rows)
    out_ids_path = args.output + ".ids"
    
    print("Aligning embeddings and IDs...")
    with open(out_ids_path, 'w') as f_ids:
        for internal_id in tqdm(range(num_docs)):
            # Get External ID for this Internal ID
            ext_id = reader.convert_internal_docid_to_collection_docid(internal_id)
            
            # Write ID to new file (Line N = ID for Row N)
            f_ids.write(ext_id + "\n")

            # Write Embedding to new file (Row N = Embedding for ID N)
            if ext_id in ext_to_row:
                row_idx = ext_to_row[ext_id]
                aligned[internal_id] = original[row_idx]
            else:
                aligned[internal_id] = np.zeros(dim, dtype=original.dtype)

    del aligned # Flush NPY to disk
    print(f"Done! \nAligned Embeddings: {args.output}\nAligned IDs: {out_ids_path}")

if __name__ == "__main__":
    main()