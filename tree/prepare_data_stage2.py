#!/usr/bin/env python3
"""
build_stage2_pir_db.py [OPTIMIZED]

Builds the "stage-2" PIR DB structure for Round 2 (sub-block bounds).
Outputs:
  - data.bin   : rows of r uint8 (one row per (term_id, leaf_node) with any non-zero subblock score)
  - idmap.bin  : fixed-size binary records (term_id:uint32, node_id:uint32, row_index:uint64)

Streaming, memory-friendly: iterates terms and writes rows as they are produced.
Requires pyserini, numpy, tqdm.

Optimizations:
1.  Vectorization: Uses NumPy to significantly speed up the per-leaf score
    calculations. The loops over `r` sub-blocks for scoring, scaling,
    and clamping are replaced with vectorized NumPy operations.
2.  I/O Buffering: Instead of writing to disk for every (term, leaf) pair,
    writes are batched into a 1MB buffer (per file) to drastically
    reduce the number of I/O syscalls.
3.  NumPy Data Structures: Uses NumPy arrays for `leaf_min_dl` and for
    storing max TFs in the `leaf_map`, leading to more efficient
    downstream processing.

Example:
  python build_stage2_pir_db_optimized.py \
    --index /path/to/pyserini-index \
    --vocab vocab.txt \
    --out-dir ./pir_stage2 \
    --B 32 --s 8 --scale 10
"""

from __future__ import annotations
import argparse
import json
import math
import struct
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass, field  # Added import

from tqdm import tqdm
import numpy as np

# Pyserini adapter (similar to your simulation)
try:
    from pyserini.index import LuceneIndexReader
    from pyserini.search.lucene import LuceneSearcher
except Exception as e:
    raise SystemExit("Pyserini is required: pip install pyserini\n" + str(e))


# ---------------- BM25 utilities (from your simulation) ----------------
@dataclass
class BM25Params:
    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1 = k1
        self.b = b

def idf(df: int, N: int) -> float:
    # same idf as in your simulation
    return math.log(1 + (N - df + 0.5) / (df + 0.5))

def bm25_term(tf: np.ndarray | int | float, dl: int, avgdl: float, p: BM25Params) -> np.ndarray | float:
    """
    Vectorized BM25 term score calculation.
    `tf` can be a scalar or a NumPy array.
    """
    safe_avgdl = max(1.0, avgdl)
    k1_plus_1 = p.k1 + 1.0
    k1_b_part = p.k1 * (1.0 - p.b)
    k1_b_dl_part = p.k1 * p.b * (dl / safe_avgdl)

    num = k1_plus_1 * tf
    denom = tf + k1_b_part + k1_b_dl_part
    
    # np.maximum works on both scalars and arrays, preventing division by zero
    return num / np.maximum(1e-9, denom)

# Tighter UB using min_dl (same form as your simulation)
def bm25_term_ub_with_min_dl(tf: np.ndarray | int | float, min_dl: int, avgdl: float, p: BM25Params) -> np.ndarray | float:
    return bm25_term(tf, min_dl, avgdl, p)


# ---------------- Lucene / Pyserini adapter ----------------

class LuceneAdapter:
    def __init__(self, index_dir: str):
        print(f"Loading Lucene index from: {index_dir}")
        self.reader = LuceneIndexReader(index_dir)
        self.searcher = LuceneSearcher(index_dir)

    @property
    def N(self) -> int:
        return int(self.reader.stats()["documents"])

    @property
    def avgdl(self) -> float:
        st = self.reader.stats()
        return float(st["total_terms"] / max(1, st["documents"]))

    # _postings_cache: Dict[str, List[Tuple[int, int]]] = {}
    def postings(self, term: str) -> List[Tuple[int, int]]:
        # if term in self._postings_cache:
        #     return self._postings_cache[term]
        pl = self.reader.get_postings_list(term, analyzer=None)
        lst = [] if pl is None else [(int(p.docid), int(p.tf)) for p in pl]
        # self._postings_cache[term] = lst
        return lst

    def df(self, term: str) -> int:
        return int(self.reader.get_term_counts(term, analyzer=None)[0])

    def analyze(self, text: str) -> List[str]:
        try:
            return [t for t in self.reader.analyze(text) if t]
        except Exception:
            try:
                from pyserini.analysis import Analyzer
                return [t for t in Analyzer().analyze(text) if t]
            except Exception:
                return [w.lower() for w in text.split() if w.strip()]

    def doc_len(self, internal_id: int) -> int:
        try:
            return int(self.reader.get_document_length_by_internal_docid(internal_id))
        except Exception: pass
        try:
            ext = self.reader.convert_internal_docid_to_collection_docid(internal_id)
            return int(self.reader.get_document_length(ext))
        except Exception: pass
        try:
            raw = self.searcher.doc_by_internal_docid(internal_id).raw()
            if raw:
                try: text = json.loads(raw).get('contents', raw)
                except Exception: text = raw
                return len(self.analyze(text))
        except Exception: pass
        return int(round(self.avgdl))

    def bm25_topk(self, query: str, k: int) -> List[Tuple[str, float]]:
        hits = self.searcher.search(query, k=k)
        # return external docids directly
        return [(h.docid, float(h.score)) for h in hits]


# ---------------- helper: layout and mapping ----------------
# ---- Layout with Pre-computed Stats -----------------------------------------
# This class now pre-computes and stores min_dl for all nodes in the conceptual tree.
@dataclass
class LayoutWithStats:
    N: int
    B: int
    r: int
    min_dl_map: Dict[int, int] = field(default_factory=dict, repr=False)

    def num_leaves(self) -> int: return math.ceil(self.N / self.B)
    def height(self) -> int: return math.ceil(math.log(self.num_leaves(), self.r)) if self.r > 1 else 0

    def node_children(self, nodes: List[int]) -> List[int]:
        return [i * self.r + j for i in nodes for j in range(self.r)]

    def node_to_doc_range(self, node: int) -> Tuple[int, int]:
        level = math.floor(math.log(max(1, node - 1), self.r)) if self.r > 1 else 0
        h = self.height()
        leaves_per_node = self.r**(h - level)
        first_node_at_level = self.r**level if self.r > 1 else 1
        idx_in_level = node - first_node_at_level
        leaf_start = idx_in_level * leaves_per_node
        doc_start = leaf_start * self.B
        doc_end = min(self.N, doc_start + leaves_per_node * self.B)
        return doc_start, doc_end


    def precompute_stats(self, adp: LuceneAdapter):
        """Iterates through all docs once to compute min_dl for all blocks."""
        print("Preprocessing: computing min_dl for all blocks...")
        doc_lengths = [adp.doc_len(i) for i in tqdm(range(self.N), desc="Fetching doc lengths")]

        num_nodes = (self.r**(self.height() + 1) - 1) // (self.r - 1) if self.r > 1 else self.height() + 1
        
        # Initialize with a large value
        self.min_dl_map = {i: float('inf') for i in range(1, num_nodes + 2)}

        leaf_start_node = self.r**self.height() if self.r > 1 else 1
        for i in tqdm(range(self.num_leaves()), desc="Computing leaf min_dl"):
            doc_start, doc_end = i * self.B, min((i + 1) * self.B, self.N)
            if doc_start < doc_end:
                leaf_min_dl = min(doc_lengths[doc_start:doc_end])
                self.min_dl_map[leaf_start_node + i] = leaf_min_dl

        # Propagate minimums up the tree
        for level in range(self.height() - 1, -1, -1):
            level_start_node = self.r**level if self.r > 1 else 1
            num_nodes_in_level = self.r**level
            for i in range(num_nodes_in_level):
                parent_node = level_start_node + i
                children = self.node_children([parent_node])
                min_child_dl = min(self.min_dl_map.get(c, float('inf')) for c in children)
                self.min_dl_map[parent_node] = min_child_dl
        print("Preprocessing complete.")

        
# ---------------- main builder ----------------
def build_stage2(
    index_dir: str,
    vocab_path: str,
    out_dir: str,
    B: int,
    s: int,
    r: int | None, # tree arity
    scale: float,
    compute_leaf_mins: bool,
    bm25_k1: float,
    bm25_b: float,
):
    # derived params
    assert B % s == 0, "B must be divisible by s"
    r_computed = B // s # number of sub-blocks per leaf block
    # if r is not None and r != r_computed:
    #     print(f"Warning: provided r={r} differs from B/s={r_computed}. Using r = B/s = {r_computed}")
    # r = r_computed
    adp = LuceneAdapter(index_dir)
    N = adp.N
    avgdl = adp.avgdl
    p = BM25Params(k1=bm25_k1, b=bm25_b)
    
    avgdl_int = int(round(avgdl))
    max_int32 = np.iinfo(np.int32).max

    num_leaves = math.ceil(N / B)
    # layout = LayoutWithStats(N=N, B=B, r=r)
    # layout.precompute_stats(adp)
    with open('data/layout_with_stats.json', 'r') as f:
            data = json.load(f)
    layout = LayoutWithStats(N=data['N'], B=data['B'], r=data['r'], min_dl_map=data['min_dl_map'])
    height = layout.height()
    # leaf_start_node = leaf_start_node_for_height(r, height)

    leaf_start_node = layout.r**height

    print(f"N={N}  B={B}  s={s}  r={layout.r}  num_leaves={num_leaves}  height={height}  leaf_start_node={leaf_start_node}")
    print(f"avgdl={avgdl:.3f}")

    # Optionally compute leaf min_dl (recommended, same as your simulation).
    # This requires one linear pass over documents but is memory-light: we keep an array of length num_leaves.
    leaf_min_dl: np.ndarray
    if compute_leaf_mins: # we're not going to do this!
        print("Computing leaf-level min_dl (one pass over documents)...")
        # Use a Python list for building, as it's faster for sparse updates
        leaf_min_dl_list = [max_int32] * num_leaves
        for docid in tqdm(range(N), desc="doc lengths"):
            dl = adp.doc_len(docid)
            leaf_idx = docid // B
            if dl < leaf_min_dl_list[leaf_idx]:
                leaf_min_dl_list[leaf_idx] = dl
        
        # replace max_int32 with avgdl if some leaf empty
        for i in range(num_leaves):
            if leaf_min_dl_list[i] == max_int32:
                leaf_min_dl_list[i] = avgdl_int
        
        # Convert to NumPy array for efficient access
        leaf_min_dl = np.array(leaf_min_dl_list, dtype=np.int32)
    else:
        print("Skipping explicit leaf min_dl computation. Will use avgdl as proxy.")
        # leaf_min_dl = np.full(num_leaves, avgdl_int, dtype=np.int32)

    # prepare outputs
    data_path = f"{out_dir.rstrip('/')}/data.bin"
    idmap_path = f"{out_dir.rstrip('/')}/idmap.bin"

    data_f = open(data_path, "wb")
    idmap_f = open(idmap_path, "wb")
    
    # --- OPTIMIZATION: Add I/O buffers ---
    data_buffer = bytearray()
    idmap_buffer = bytearray()
    # Flush buffers when they exceed 1MB (tune as needed)
    BUFFER_FLUSH_SIZE = 1024 * 1024 * 1  # 1MB

    row_index = 0  # 64-bit

    # prepare vocab iteration
    # simple vocab file: one term per line -> term_id is file line order starting at 1
    print(f"Reading vocab from: {vocab_path}")
    # terms = []
    terms = json.load(open(vocab_path, 'r', encoding='utf-8'))
    # for term, term_id in vocab.items():
    #     terms.append(term)
    # terms.sort(key=lambda t: vocab[t])  # sort by term_id
    # with open(vocab_path, 'r', encoding='utf-8') as vf:
    #     terms = [line.rstrip('\n') for line in vf if line.strip()]
    i = 0
    print(f"Vocab contains {len(terms)} terms. Processing terms streaming...")
    for term, term_id in (pbar := tqdm(terms.items(), desc="terms")):
        # if i < 1050766: # continue from last 
        #     i += 1
        #     continue
        
        # term_id = term_id_zero + 1  # 1-based term_id (you can change)
        # term = adp.analyze(term_id)[0]  # get the actual term string
        term_id = int(term_id)
        # print(term, term_id)
        postings = adp.postings(term)
        if not postings:
            continue

        df_term = len(postings)
        term_idf = idf(df_term, N)
        if term_idf <= 0:
            # unlikely, skip
            continue

        # For this term we will build a map:
        #   leaf_node -> array[max_tf_per_subblock] length r
        
        # --- OPTIMIZATION: Use defaultdict and NumPy array for map values ---
        leaf_map: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(r_computed, dtype=np.int32))
        
        for docid, tf in postings:
            leaf_idx = docid // B
            leaf_node = leaf_start_node + leaf_idx
            within_leaf = docid % B
            subblock_idx = within_leaf // s  # 0 .. r-1

            # Get (or create) the array for this leaf
            arr = leaf_map[leaf_node]
            if tf > arr[subblock_idx]:
                arr[subblock_idx] = tf

        # for each leaf_node produce the r subblock ub scores (float), scale+round to uint8
        for leaf_node, max_tfs in leaf_map.items():
            # get leaf-level min_dl
            # leaf_idx = leaf_node - leaf_start_node
            
            min_dl_for_leaf = layout.min_dl_map.get(leaf_node, avgdl_int)
            # --- OPTIMIZATION: Vectorized score computation ---
            # We only want to compute for tf_sub > 0
            nonzero_mask = max_tfs > 0
            
            # Start with an array of zeros
            subblock_scores_float = np.zeros(r_computed, dtype=np.float64)
            
            any_nonzero = np.any(nonzero_mask)

            if any_nonzero:
                # Only compute scores for sub-blocks that had the term
                active_tfs = max_tfs[nonzero_mask]
                scores = bm25_term_ub_with_min_dl(active_tfs, min_dl_for_leaf, avgdl, p) * term_idf
                # Place computed scores back into the full array
                subblock_scores_float[nonzero_mask] = scores
            
            if not any_nonzero:
                # skip this (term,leaf) since all subblocks zero
                continue

            # --- OPTIMIZATION: Vectorized scaling, clamping, and packing ---
            # Scale, round, clip to [0, 255], and convert to uint8 in one go
            scaled_scores = np.round(subblock_scores_float * scale)
            packed_np = np.clip(scaled_scores, 0, 255).astype(np.uint8)
            
            # Convert the NumPy array of r uint8s to r raw bytes
            packed = packed_np.tobytes()

            # The `if len(packed) != r:` check is no longer needed
            # as `packed_np` is guaranteed to have length `r`.

            # --- OPTIMIZATION: Write to buffers instead of files ---
            
            # write data row (r bytes)
            data_buffer.extend(packed)

            # write idmap record: term_id:uint32, node_id:uint32, row_index:uint64 (little endian)
            idmap_buffer.extend(struct.pack("<IIQ", term_id, int(leaf_node), int(row_index)))

            row_index += 1
        pbar.set_postfix({"rows": row_index})
            
        # --- OPTIMIZATION: Flush buffers if they are full (inside term loop) ---
        if len(data_buffer) > BUFFER_FLUSH_SIZE:
            print(f"Flushing buffers at term_id={term_id}, row_index={row_index}...")
            data_f.write(data_buffer)
            idmap_f.write(idmap_buffer)
            data_buffer.clear()
            idmap_buffer.clear()

    # --- OPTIMIZATION: Final flush for any remaining data ---
    if data_buffer:
        data_f.write(data_buffer)
        idmap_f.write(idmap_buffer)
        data_buffer.clear()
        idmap_buffer.clear()

    data_f.close()
    idmap_f.close()

    print(f"Done. Wrote {row_index} rows.")
    print(f"data.bin -> {data_path}")
    print(f"idmap.bin -> {idmap_path}")


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Build stage-2 PIR DB (sub-block bounds) streaming [OPTIMIZED]")
    ap.add_argument("--index", required=True, help="Path to Pyserini/Lucene index")
    ap.add_argument("--vocab", required=True, help="Vocab file: one term per line")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--B", type=int, default=32, help="Block size (docs per leaf block)")
    ap.add_argument("--s", type=int, default=8, help="Sub-block size (docs per sub-block)")
    ap.add_argument("--scale", type=float, default=10.0, help="Scale factor when converting float->uint8")
    ap.add_argument("--compute-leaf-mins", action="store_true",
                      help="Compute leaf-level min_dl by scanning docs (recommended). If not set, uses avgdl")
    ap.add_argument("--bm25-k1", type=float, default=0.9)
    ap.add_argument("--bm25-b", type=float, default=0.4)
    # optional r parameter (we compute r = B/s and warn if user provided different)
    ap.add_argument("--r", type=int, default=None, help="(optional) Arity; defaults to B/s")
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    build_stage2(
        index_dir=args.index,
        vocab_path=args.vocab,
        out_dir=args.out_dir,
        B=args.B,
        s=args.s,
        r=args.r,
        scale=args.scale,
        compute_leaf_mins=args.compute_leaf_mins,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
    )

if __name__ == "__main__":
    main()
