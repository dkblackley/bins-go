#!/usr/bin/env python3
"""
Create document mapping file (docmap.bin) from a Lucene/Pyserini index.
This maps internal document IDs to their external IDs.
"""

import os
import struct
from pyserini.search.lucene import LuceneSearcher
from pyserini.index import LuceneIndexReader
from tqdm import tqdm

def create_docmap(index_path: str, output_path: str):
    """
    Create a binary file mapping internal docids to external docids.
    Format: uint64 for each mapping, little-endian
    """
    print(f"Loading index from: {index_path}")
    searcher = LuceneSearcher(index_path)
    indexer = LuceneIndexReader(index_path)
    # Get all documents by searching for *:*
    # hits = searcher.search("*:*", k=searcher.num_docs)
    # get number of documents
    num_docs = int(indexer.stats()["documents"])
    # get all document ids
    doc_ids = list(range(num_docs))
    # get external ids
    ext_ids = [indexer.convert_internal_docid_to_collection_docid(i) for i in doc_ids]
    print(f"Found {len(ext_ids)} documents")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        for hit in tqdm(ext_ids, desc="Creating document mapping"):
            try:
                # Get external docid from hit and convert to integer
                external_id = hit
                # Some collections might use string IDs, try to convert to int
                try:
                    external_id = int(external_id)
                except ValueError:
                    # If it's not a number, hash it to get a stable integer
                    external_id = hash(external_id) & ((1 << 64) - 1)  # Ensure it fits in uint64
                
                # Write as 8-byte little-endian unsigned long long
                f.write(struct.pack('<Q', external_id))
            except Exception as e:
                print(f"Error processing docid {hit}: {e}")
                # Write 0 for failed conversions
                f.write(struct.pack('<Q', 0))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create document mapping file from Lucene index')
    parser.add_argument('--index', required=True, help='Path to Lucene index')
    parser.add_argument('--output', required=True, help='Path to output docmap.bin file')
    args = parser.parse_args()
    
    create_docmap(args.index, args.output)