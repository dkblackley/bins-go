# preprocess queries to the 192-dim msmarco-MiniLM-L-6-v3 embeddings with PCA

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import faiss
import argparse
from numpy.lib.format import open_memmap

import json

def load_queries(queries_path):
    """Loads queries into memory, auto-detecting TSV vs JSONL."""
    is_jsonl = queries_path.endswith('.jsonl')
    queries = {}

    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            if is_jsonl:
                # Handle BEIR JSONL format
                data = json.loads(line)
                queries[data['_id']] = f"query: {data.get('text', '').strip()}"
            else:
                # Handle standard TSV format
                qid, qtext = line.rstrip('\n').split('\t', 1)
                queries[qid] = f"query: {qtext}"

    return queries

def iter_queries(queries, batch_size=2048):
    buf_ids, buf_txt = [], []
    for qid, qtext in queries.items():
        buf_ids.append(qid)
        buf_txt.append(qtext)
        if len(buf_ids) == batch_size:
            yield buf_ids, buf_txt
            buf_ids, buf_txt = [], []
    if buf_ids:
        yield buf_ids, buf_txt

def main(args):
    queries = load_queries(args.queries)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3', device=device)
    model = SentenceTransformer('intfloat/e5-base-v2', device=device)
    query_ids = list(queries.keys())
    out = open_memmap(args.output, mode='w+', dtype=np.float32, shape=(len(query_ids), 192))
    # pca = faiss.read_VectorTransform("../../../datasets/Son/pca_768_to_192.faiss")
    pca = faiss.read_VectorTransform("pca_e5_768_to_192.faiss")
    batch_size = 256  # Increased for efficiency
    for i, (buf_ids, buf_txt) in enumerate(tqdm(iter_queries(queries, batch_size=batch_size))):
        embeddings = model.encode(buf_txt, convert_to_numpy=True, normalize_embeddings=True)
        embeddings_pca = pca.apply_py(embeddings)
        start_idx = i * batch_size
        out[start_idx:start_idx + len(buf_ids), :] = embeddings_pca
    del out  # flush to disk
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=str, required=True, help='Path to queries.tsv')
    parser.add_argument('--output', type=str, required=True, help='Path to output .npy file for query embeddings')
    args = parser.parse_args()
    main(args)