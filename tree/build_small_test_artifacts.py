#!/usr/bin/env python3
# Usage: python build_small_test_artifacts.py <dataset dir with corpus.jsonl/queries.jsonl/qrels/test.tsv/*_corpus_embed_f32.npy(.ids)> <out dir> [B r s [name]]
# name defaults to scifact; doc IDs must be integers (datasets/prepare_dataset.py renumbers them 0..N-1).
# For a whole new dataset use datasets/prep_tree.sh, which builds the index and calls this.
# Needs a single-threaded pyserini index at <out dir>/index_<name> (so internal docids follow corpus order):
#   python -m pyserini.index.lucene --collection JsonCollection --input <coll dir> --index <out dir>/index_scifact --generator DefaultLuceneDocumentGenerator --threads 1 --storeDocvectors
# Small-scale port of the collaborator's prep scripts (prepare_data.py, prepare_data_stage2.py,
# gen_layout.py, tools/create_docmap.py and opt3.py's query analysis) for a scifact smoke test.
import json, math, struct, sys, os, numpy as np
from pyserini.index.lucene import LuceneIndexReader
from pyserini.analysis import Analyzer, get_lucene_analyzer
D, T, B, r, s, scale, k1, b = sys.argv[1], sys.argv[2], 8, 4, 4, 10.0, 0.9, 0.4
if len(sys.argv) > 3: B, r, s = map(int, sys.argv[3:6])
name = sys.argv[6] if len(sys.argv) > 6 else "scifact"
idx = f"{T}/index_{name}"
rd = LuceneIndexReader(idx)
st = rd.stats(); N = int(st["documents"]); avgdl = st["total_terms"] / N
num_leaves = math.ceil(N / B); height = math.ceil(math.log(num_leaves, r)) if num_leaves > 1 else 0
out = f"{T}/B{B}_r{r}_s{s}"; os.makedirs(f"{out}/stage2", exist_ok=True)
json.dump({"N": N, "B": B, "r": r, "min_dl_map": {}}, open(f"{out}/layout.json", "w"))
idf = lambda df: math.log(1 + (N - df + 0.5) / (df + 0.5))
ub = lambda tf, dl: (k1 + 1) * tf / (tf + k1 * (1 - b + b * dl / max(1.0, avgdl)))
terms = [t.term for t in rd.terms()]
vocab = {t: str(i) for i, t in enumerate(terms)}
json.dump(vocab, open(f"{out}/stage1_vocab.json", "w"))
d1, m1 = open(f"{out}/stage1_data.bin", "wb"), open(f"{out}/stage1_idmap.bin", "wb")
d2, m2 = open(f"{out}/stage2/data.bin", "wb"), open(f"{out}/stage2/idmap.bin", "wb")
row1 = row2 = 0; nsub = B // s; leaf_start = r ** height; ia = int(avgdl); iar = int(round(avgdl))
for tid, term in enumerate(terms):
    pl = rd.get_postings_list(term, analyzer=None)
    if not pl: continue
    post = [(p.docid, p.tf) for p in pl]; tidf = idf(len(post))
    # ---- stage1 (prepare_data.py) ----
    cur = np.zeros(num_leaves, dtype=np.int32)
    for d, tf in post: cur[d // B] = max(cur[d // B], tf)
    for level in range(height - 1, -1, -1):
        lpn = r ** (height - level); nodes = math.ceil(num_leaves / lpn); start = r ** level
        pad = int(math.ceil(cur.shape[0] / r) * r); pm = np.zeros(pad, dtype=np.int32); pm[:cur.shape[0]] = cur
        cm = pm.reshape(-1, r)[:nodes]
        for ni in np.where((cm > 0).any(axis=1))[0]:
            sc = np.array([ub(int(t), ia) * tidf if t > 0 else 0.0 for t in cm[ni]])
            d1.write(np.clip(np.round(sc * scale), 0, 255).astype(np.uint8).tobytes())
            m1.write(struct.pack("<IIQ", tid, start + int(ni), row1)); row1 += 1
        cur = cm.max(axis=1)
    # ---- stage2 (prepare_data_stage2.py) ----
    leaf = {}
    for d, tf in post:
        a = leaf.setdefault(leaf_start + d // B, np.zeros(nsub, dtype=np.int32)); j = (d % B) // s; a[j] = max(a[j], tf)
    for ln, a in leaf.items():
        sc = np.array([ub(int(t), iar) * tidf if t > 0 else 0.0 for t in a])
        d2.write(np.clip(np.round(sc * scale), 0, 255).astype(np.uint8).tobytes())
        m2.write(struct.pack("<IIQ", tid, ln, row2)); row2 += 1
for f in (d1, m1, d2, m2): f.close()
print(f"N={N} height={height} vocab={len(terms)} stage1 rows={row1} stage2 rows={row2}")
# ---- docmap.bin (tools/create_docmap.py) ----
ext = [rd.convert_internal_docid_to_collection_docid(i) for i in range(N)]
with open(f"{idx}/docmap.bin", "wb") as f:
    for e in ext: f.write(struct.pack("<Q", int(e)))
# ---- stage3 doc embeddings in index order + ids (only written once per dataset, shared by every B/r/s) ----
if not os.path.exists(f"{T}/stage3_doc.npy.ids"):
    ids = [l.strip() for l in open(f"{D}/{name}_corpus_embed_f32.npy.ids")]; pos = {x: i for i, x in enumerate(ids)}
    Y = np.load(f"{D}/{name}_corpus_embed_f32.npy", mmap_mode="r")
    np.save(f"{T}/stage3_doc.npy", Y[[pos[e] for e in ext]].astype(np.float32))
    open(f"{T}/stage3_doc.npy.ids", "w").write("\n".join(ext) + "\n")
# ---- queries.tsv + analyzed json (opt3.py), only queries that have test qrels ----
qrels = [l.split("\t") for l in open(f"{D}/qrels/test.tsv").read().splitlines()[1:]]
want = {q[0] for q in qrels}
an = Analyzer(get_lucene_analyzer())
qs = [json.loads(l) for l in open(f"{D}/queries.jsonl")]
qs = [q for q in qs if q["_id"] in want]
with open(f"{T}/queries.qrels.tsv", "w") as f:
    for q in qs: f.write(f"{q['_id']}\t{q['text']}\n")
json.dump([{"id": q["_id"], "text": q["text"], "terms": an.analyze(q["text"])} for q in qs], open(f"{T}/queries.qrels.analyzed.json", "w"))
with open(f"{T}/qrels.trec", "w") as f:
    for q in qrels: f.write(f"{q[0]} 0 {q[1]} {q[2]}\n")
print("queries", len(qs))
