# first process all ms marco documents to vector embeddings
# then use PCA to reduce dimensions to 192
def load_documents(documents_path):
    # documents_path is collection.tsv
    documents = {}
    with open(documents_path, 'r') as f:
        for line in f:
            docid, doctext = line.strip().split('\t', 1)
            documents[docid] = doctext
    return documents
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
# def main(args):
#     documents = load_documents(args.documents)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3', device=device)
#     embeddings = {}
#     # batching for efficiency
#     batch_size = 2048
#     docids = list(documents.keys())
#     all_embeddings = []
#     for i in tqdm(range(0, len(docids), batch_size)):
#         batch_docids = docids[i:i+batch_size]
#         batch_texts = [documents[docid] for docid in batch_docids]
#         batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
#         # for j, docid in enumerate(batch_docids):
#         all_embeddings.extend(batch_embeddings)
#     np.save(args.output, np.array(all_embeddings))

import numpy as np
from numpy.lib.format import open_memmap
from sentence_transformers import SentenceTransformer
import torch
import faiss
from tqdm import tqdm

def iter_docs(path, batch_size=2048):
    buf_ids, buf_txt = [], []
    with open(path, 'r') as f:
        for line in f:
            docid, doctext = line.rstrip('\n').split('\t', 1)
            buf_ids.append(docid)
            buf_txt.append(doctext)
            if len(buf_ids) == batch_size:
                yield buf_ids, buf_txt
                buf_ids, buf_txt = [], []
    if buf_ids:
        yield buf_ids, buf_txt

def count_lines(path):
    print(f"Counting lines in {path}...")
    with open(path, 'r') as f:
        return sum(1 for _ in f)

def main(args):
    # 1) Count first so we can preallocate on disk
    num_docs = count_lines(args.documents)
    print(f"Total documents: {num_docs:,}")

    output_dim = 192  # After PCA reduction
    dtype = np.float32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3', device=device)

    # Load PCA transform
    print("Loading PCA transform...")
    pca = faiss.read_VectorTransform("../datasets/Son/pca_768_to_192.faiss")
    print(f"PCA: {pca.d_in} -> {pca.d_out} dimensions")

    # 2) Preallocate a memmapped .npy file for 192-dim embeddings
    out = open_memmap(args.output, mode='w+', dtype=dtype, shape=(num_docs, output_dim))
    ids_path = args.output + '.ids'  # save mapping
    with open(ids_path, 'w') as fid:
        row = 0
        for docids, texts in tqdm(iter_docs(args.documents, batch_size=2048),
                                   total=(num_docs + 2047) // 2048,
                                   desc="Processing documents"):
            embs = model.encode(texts, convert_to_numpy=True)  # (B, 384) float32
            embs_pca = pca.apply_py(embs)  # (B, 192) float32
            n = len(docids)
            out[row:row+n] = embs_pca
            for d in docids:
                fid.write(d + '\n')
            row += n

    print(f"\nProcessed {row:,} documents")
    print(f"Output saved to: {args.output}")
    print(f"ID mapping saved to: {ids_path}")

    # Ensure flush to disk
    del out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--documents', type=str, required=True, help='Path to collection.tsv')
    ap.add_argument('--output', type=str, required=True, help='Path to output embeddings file')
    args = ap.parse_args()
    main(args)