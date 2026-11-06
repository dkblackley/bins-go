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

import json

def iter_docs(path, batch_size=2048):
    """Yields batches of documents, auto-detecting TSV vs JSONL."""
    is_jsonl = path.endswith('.jsonl')
    buf_ids, buf_txt = [], []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if is_jsonl:
                # Handle BEIR JSONL format
                data = json.loads(line)
                docid = data['_id']
                title = data.get('title', '')
                text = data.get('text', '')
                doctext = f"passage: {title} {text}".strip()
            else:
                # Handle standard TSV format
                docid, doctext = line.rstrip('\n').split('\t', 1)
                doctext = f"passage: {doctext}"

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

def make_pca(model, docs_path, sample_size=100000, batch_size=2048, pca_path="pca_e5_768_to_192.faiss"):
     """Dynamically trains a 768 -> 192 PCA matrix using a sample of documents."""
     print(f"\n--- Training PCA Transform ---")
     print(f"Collecting ~{sample_size} documents for PCA training...")
     sample_texts = []

     for _, texts in iter_docs(docs_path, batch_size=batch_size):
         sample_texts.extend(texts)
         if len(sample_texts) >= sample_size:
             break

     # Trim to exact sample size if we overshot via batching
     sample_texts = sample_texts[:sample_size]

     print(f"Encoding {len(sample_texts)} documents for PCA...")
     # E5 requirement: normalize_embeddings=True
     embeddings_sample = model.encode(sample_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size)

     print("Training FAISS PCA Matrix (768 -> 192)...")
     pca = faiss.PCAMatrix(768, 192)
     pca.train(embeddings_sample)

     # Save the matrix to disk for the query script. Use a per-dataset --pca path, otherwise embedding a second
     # dataset overwrites the PCA that the first dataset's queries need.
     faiss.write_VectorTransform(pca, pca_path)
     print(f"PCA training complete and saved to {pca_path}!\n")
     return pca

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3', device=device)
    model = SentenceTransformer('intfloat/e5-base-v2', device=device)
    if args.fp16 and device == 'cuda':
        model = model.half()  # ~2x faster on A100; only worth it for the 100M+ corpora

    # PCA: either load an existing one (sharded runs) or train it on a sample of this corpus
    if args.load_pca:
        print(f"Loading PCA transform from {args.pca}")
        pca = faiss.read_VectorTransform(args.pca)
    else:
        pca = make_pca(model, args.documents, sample_size=args.pca_sample, batch_size=32, pca_path=args.pca)
    print(f"PCA: {pca.d_in} -> {pca.d_out} dimensions")
    if args.pca_only:
        return

    # 1) Count first so we can preallocate on disk
    num_docs = args.num_docs if args.num_docs > 0 else count_lines(args.documents)
    print(f"Total documents: {num_docs:,}")

    # Optional sharding (for 100M+ corpora): shard i of n embeds docs [lo, hi) into its own .npy. Merge the shards
    # afterwards with datasets/merge_embed_shards.py.
    per = (num_docs + args.num_shards - 1) // args.num_shards
    lo, hi = args.shard_id * per, min(num_docs, (args.shard_id + 1) * per)
    out_path = args.output
    if args.num_shards > 1:
        out_path = f"{args.output}.shard{args.shard_id:03d}of{args.num_shards:03d}.npy"
        print(f"Shard {args.shard_id}/{args.num_shards}: docs [{lo:,}, {hi:,}) -> {out_path}")

    output_dim = pca.d_out
    dtype = np.float32
    batch = 2048

    # 2) Preallocate a memmapped .npy file for 192-dim embeddings
    out = open_memmap(out_path, mode='w+', dtype=dtype, shape=(hi - lo, output_dim))
    ids_path = out_path + '.ids'  # save mapping
    with open(ids_path, 'w') as fid:
        row = 0
        seen = 0
        for docids, texts in tqdm(iter_docs(args.documents, batch_size=batch),
                                   total=(num_docs + batch - 1) // batch,
                                   desc="Processing documents"):
            # skip batches before this shard / stop after it (batches straddling an edge are trimmed)
            b_lo, b_hi = seen, seen + len(docids)
            seen = b_hi
            if b_hi <= lo:
                continue
            if b_lo >= hi:
                break
            s0, s1 = max(lo, b_lo) - b_lo, min(hi, b_hi) - b_lo
            docids, texts = docids[s0:s1], texts[s0:s1]

            embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                batch_size=args.encode_batch)  # (B, 768)
            embs_pca = pca.apply_py(np.ascontiguousarray(embs, dtype=np.float32))  # (B, 192) float32
            n = len(docids)
            out[row:row+n] = embs_pca
            for d in docids:
                fid.write(d + '\n')
            row += n

    assert row == hi - lo, f"wrote {row} rows, expected {hi - lo} (wrong --num-docs?)"
    print(f"\nProcessed {row:,} documents")
    print(f"Output saved to: {out_path}")
    print(f"ID mapping saved to: {ids_path}")

    # Ensure flush to disk
    del out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--documents', type=str, required=True, help='Path to collection.tsv or BEIR corpus.jsonl')
    ap.add_argument('--output', type=str, required=True, help='Path to output embeddings file')
    ap.add_argument('--pca', type=str, default='pca_e5_768_to_192.faiss',
                    help='Where to save (or, with --load-pca, read) the 768->192 PCA. Use one per dataset.')
    ap.add_argument('--load-pca', action='store_true', help='Reuse an existing --pca instead of training one')
    ap.add_argument('--pca-only', action='store_true', help='Only train/save the PCA, then exit')
    ap.add_argument('--pca-sample', type=int, default=100000, help='Docs used to train the PCA')
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard-id', type=int, default=0)
    ap.add_argument('--num-docs', type=int, default=0, help='Skip the line count if you already know it')
    ap.add_argument('--encode-batch', type=int, default=32, help='SentenceTransformer encode batch size')
    ap.add_argument('--fp16', action='store_true', help='Encode in half precision (GPU only)')
    args = ap.parse_args()
    main(args)
