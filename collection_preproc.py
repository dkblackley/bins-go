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
import os
import sys

# Exit code when another job (e.g. the same embed queued on a second GPU partition) already finished this output.
# Non-zero on purpose: the jobs that depend on THIS job (afterok) then never start, so two racing series don't both
# launch their tree-db / grid jobs on the same files. datasets/README.md "Racing two GPU queues" has the details.
ALREADY_DONE = 3


def done_marker(path):
    return path + '.done'


def is_done(path):
    return os.path.exists(done_marker(path))


def claim_done(path):
    """Atomically create <path>.done; False if some other job got there first."""
    try:
        os.close(os.open(done_marker(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def remove_quiet(*paths):
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def enable_fast_fp32():
    # TF32 matmuls: ~3-8x faster fp32 on A100/H100/B200 (PyTorch leaves it off by default), the e5/PCA vectors are
    # unchanged to ~1e-3. EMBED_TF32=0 turns it off.
    if os.environ.get('EMBED_TF32', '1') != '0':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
     # dataset overwrites the PCA that the first dataset's queries need. Written to a temp file and hard-linked into
     # place, so if another job saved one meanwhile we use theirs and every shard/query shares one PCA.
     tmp = f"{pca_path}.tmp{os.getpid()}"
     faiss.write_VectorTransform(pca, tmp)
     try:
         os.link(tmp, pca_path)
         print(f"PCA training complete and saved to {pca_path}!\n")
     except FileExistsError:
         print(f"Another job saved {pca_path} first; using that one")
         pca = faiss.read_VectorTransform(pca_path)
     os.remove(tmp)
     return pca

def main(args):
    if args.overwrite and args.num_shards == 1:   # shards: --overwrite only redoes this shard (see below)
        remove_quiet(done_marker(args.output), args.pca)
    if is_done(args.output):
        print(f"{args.output} is already complete ({done_marker(args.output)} exists, another job finished it). "
              f"Pass --overwrite to redo it.")
        sys.exit(0 if args.skip_if_done else ALREADY_DONE)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    enable_fast_fp32()
    # model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3', device=device)
    model = SentenceTransformer('intfloat/e5-base-v2', device=device)
    if args.fp16 and device == 'cuda':
        model = model.half()  # tensor cores: ~5x the fp32 speed on A100, ~2x on top of TF32

    # PCA: either load an existing one (sharded runs) or train it on a sample of this corpus
    if args.load_pca or os.path.exists(args.pca):
        # an existing PCA is always reused (same model + same 100k sample), so a second job racing this one can't
        # swap the PCA under shards/queries that already used it. --overwrite retrains.
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

    if args.num_shards > 1 and args.overwrite:
        remove_quiet(done_marker(out_path))
    if args.num_shards > 1 and is_done(out_path):
        print(f"{out_path} already done by another job, skipping this shard")
        return

    output_dim = pca.d_out
    dtype = np.float32
    batch = 2048

    # 2) Write into a per-job partial file and move it into place only when complete. The partial name carries
    # EMBED_TAG (set per series by run_all.sh, else the slurm job id), so a job that is preempted + requeued, or
    # resubmitted with the same tag, resumes from its last checkpoint instead of starting over.
    tag = os.environ.get('EMBED_TAG') or os.environ.get('SLURM_JOB_ID') or 'local'
    if os.environ.get('SLURM_ARRAY_TASK_ID') and not os.environ.get('EMBED_TAG'):
        tag = f"{os.environ.get('SLURM_ARRAY_JOB_ID', tag)}_{os.environ['SLURM_ARRAY_TASK_ID']}"
    part_path = f"{out_path}.partial-{tag}.npy"
    part_ids, progress = part_path + '.ids', part_path + '.progress'
    total = hi - lo

    row = 0
    if os.path.exists(progress) and os.path.exists(part_path) and os.path.exists(part_ids):
        row = int(open(progress).read().strip() or 0)
        out = open_memmap(part_path, mode='r+')
        assert out.shape == (total, output_dim), f"{part_path} has shape {out.shape}, expected {(total, output_dim)}"
        with open(part_ids) as f:                      # drop ids written after the last checkpoint
            kept = [next(f) for _ in range(row)]
        with open(part_ids, 'w') as f:
            f.writelines(kept)
        del kept
        print(f"Resuming {part_path} from row {row:,} of {total:,}")
    else:
        out = open_memmap(part_path, mode='w+', dtype=dtype, shape=(total, output_dim))
        open(part_ids, 'w').close()

    def checkpoint(fid):
        out.flush()
        fid.flush()
        os.fsync(fid.fileno())
        with open(progress + '.tmp', 'w') as f:
            f.write(str(row))
        os.replace(progress + '.tmp', progress)

    start = lo + row      # first document still to embed
    every = max(1, args.checkpoint_every)
    with open(part_ids, 'a') as fid:
        seen = 0
        n_batches = 0
        for docids, texts in tqdm(iter_docs(args.documents, batch_size=batch),
                                   total=(num_docs + batch - 1) // batch,
                                   desc="Processing documents"):
            # skip batches before this shard / already embedded, stop after the shard (edge batches are trimmed)
            b_lo, b_hi = seen, seen + len(docids)
            seen = b_hi
            if b_hi <= start:
                continue
            if b_lo >= hi:
                break
            s0, s1 = max(start, b_lo) - b_lo, min(hi, b_hi) - b_lo
            docids, texts = docids[s0:s1], texts[s0:s1]

            embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                batch_size=args.encode_batch)  # (B, 768)
            embs_pca = pca.apply_py(np.ascontiguousarray(embs, dtype=np.float32))  # (B, 192) float32
            n = len(docids)
            out[row:row+n] = embs_pca
            for d in docids:
                fid.write(d + '\n')
            row += n
            n_batches += 1
            if n_batches % every == 0:
                checkpoint(fid)
        checkpoint(fid)

    assert row == total, f"wrote {row} rows, expected {total} (wrong --num-docs?)"
    del out  # flush to disk

    # 3) Publish. Exactly one job wins the .done marker; a loser throws its copy away.
    final_done = is_done(args.output) and args.num_shards > 1   # whole corpus already merged by another series
    if final_done or not claim_done(out_path):
        remove_quiet(part_path, part_ids, progress)
        who = args.output if final_done else out_path
        print(f"Another job finished {who} first; discarding this copy")
        if args.num_shards > 1 and not final_done:
            return            # the other series' shard is identical (same PCA): carry on to the merge
        sys.exit(0 if args.skip_if_done else ALREADY_DONE)
    os.replace(part_ids, out_path + '.ids')
    os.replace(part_path, out_path)
    remove_quiet(progress)
    print(f"\nProcessed {row:,} documents")
    print(f"Output saved to: {out_path}")
    print(f"ID mapping saved to: {out_path}.ids")


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
    ap.add_argument('--checkpoint-every', type=int, default=50,
                    help='Batches (of 2048 docs) between resume checkpoints')
    ap.add_argument('--overwrite', action='store_true',
                    help='Redo a finished output (deletes its .done marker and the PCA first)')
    ap.add_argument('--skip-if-done', action='store_true',
                    help=f'Exit 0 instead of {ALREADY_DONE} when the output is already complete')
    args = ap.parse_args()
    main(args)
