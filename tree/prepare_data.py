#!/usr/bin/env python3
"""
build_stage1_pir_db.py

Builds the "stage-1" PIR DB structure for the privacy-preserving retrieval protocol.
This is an optimized version.

Outputs (streaming):
  - data.bin   : rows of r uint8 (one row per (term,node) with any non-zero child)
  - idmap.bin  : fixed-size binary records (term_id:uint32, node_id:uint32, row_index:uint64)
  - vocab.json : small mapping term -> term_id

Requirements:
  pip install pyserini numpy tqdm

Notes:
  - You must provide a vocabulary file (--vocab) with one token per line OR
    supply --terms-from-index to try to enumerate terms from the Lucene index
    (that option may be slower / depend on your pyserini version).
  - Provide an existing data/layout_with_stats.json (from your preprocessing step)
    if you have precomputed min_dl per node; otherwise the script will use avgdl.
"""

from __future__ import annotations
import argparse
import json
import math
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Pyserini
try:
    from pyserini.index import LuceneIndexReader
except Exception as e:
    raise SystemExit("pyserini is required (pip install pyserini). Error: " + str(e))


# -------------------------
# BM25 helpers
# -------------------------

def idf(df: int, N: int) -> float:
    """Calculates the IDF score for a given document frequency."""
    return math.log(1 + (N - df + 0.5) / (df + 0.5))

def bm25_term_ub(tf: int, dl: int, avgdl: float, k1: float = 0.9, b: float = 0.4) -> float:
    """
    Calculates the BM25 term score.
    Used for upper-bounding by passing min_dl for the 'dl' parameter.
    """
    num = (k1 + 1.0) * tf
    denom = tf + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
    return num / max(1e-9, denom)

# -------------------------
# Layout helpers
# -------------------------
class Layout:
    """Helper class to manage tree layout geometry."""
    def __init__(self, N: int, B: int, r: int, min_dl_map: Dict[int,int] = None):
        self.N = N
        self.B = B
        self.r = r
        if self.r <= 1:
            raise ValueError("Fanout 'r' must be > 1")
        self.min_dl_map = min_dl_map or {}
        self.num_leaves = math.ceil(self.N / self.B)
        self.height = math.ceil(math.log(self.num_leaves, self.r)) if self.num_leaves > 1 else 0

    def node_level_start(self, level: int) -> int:
        """Returns the ID of the first node at a given level."""
        return self.r**level

    def nodes_at_level(self, level:int) -> int:
        """Returns the count of nodes at a given level."""
        return self.r**level

    def actual_nodes_at_level(self, level: int) -> int:
        """Returns the *actual* number of nodes at a given level for an incomplete tree."""
        if level == self.height:
            return self.num_leaves
        leaves_per_node_at_level = self.r**(self.height - level)
        return math.ceil(self.num_leaves / leaves_per_node_at_level)

# -------------------------
# Pyserini adapter (simplified)
# -------------------------
class SimpleReader:
    """Wrapper for Pyserini LuceneIndexReader."""
    def __init__(self, index_dir: str):
        self.reader = LuceneIndexReader(index_dir)
        stats = self.reader.stats()
        self.N = int(stats["documents"])
        self.avgdl = float(stats["total_terms"]) / max(1, int(stats["documents"]))

    def postings(self, term: str) -> List[Tuple[int,int]]:
        """Return list of (internal_docid, tf) for the term."""
        pl = self.reader.get_postings_list(term, analyzer=None)
        if pl is None:
            return []
        return [(int(p.docid), int(p.tf)) for p in pl]

    def df(self, term: str) -> int:
        """Return document frequency for the term."""
        try:
            return int(self.reader.get_term_counts(term, analyzer=None)[0])
        except Exception:
            # Catch broad exception in case term doesn't exist or other pyserini error
            return 0

# -------------------------
# Main builder
# -------------------------
def build(args):
    index = args.index
    vocab_file = args.vocab
    out_prefix = args.out_prefix
    B = args.B
    r = args.r
    scale = args.scale
    k1 = args.k1
    bparam = args.bparam

    reader = SimpleReader(index)
    N = reader.N
    avgdl = reader.avgdl
    layout = Layout(N=N, B=B, r=r)
    int_avgdl = int(avgdl)

    # optionally load precomputed data/layout_with_stats.json to get min_dl_map
    if args.layout_stats:
        print(f"Loading layout stats from: {args.layout_stats}")
        with open(args.layout_stats, 'r', encoding='utf-8') as f:
            L = json.load(f)
            layout.min_dl_map = {k:v for k, v in L.get("min_dl_map", {}).items()}
        print(f"Loaded {len(layout.min_dl_map)} min_dl entries.")

    # build vocabulary list
    if vocab_file:
        print("Loading vocabulary from:", vocab_file)
        terms = []
        with open(vocab_file, 'r', encoding='utf-8') as vf:
            for line in vf:
                t = line.strip()
                if t: terms.append(t)
    elif args.terms_from_index:
        # try to enumerate index terms (may be slow and pyserini version dependent)
        print("Enumerating terms from the index (this may be slow)...")
        terms = []
        it = reader.reader.object.getTerms(reader.reader.reader)
        for t in tqdm(it, desc="Loading vocab from index"):
            if t: terms.append(t)
    else:
        raise SystemExit("You must supply --vocab or --terms-from-index")

    print(f"Index N={N}, avgdl={avgdl:.2f}, #terms={len(terms)}, r={r}, B={B}, height={layout.height}")

    # create vocab -> id
    vocab = {t.getTerm(): idx for idx, t in enumerate(terms)}
    vocab_path = out_prefix + "_vocab.json"
    # with open(vocab_path, 'w', encoding='utf-8') as vf:
        # json.dump(vocab, vf)#, indent=2, ensure_ascii=False)
    # assert False, "Done vocab"
    data_path = Path(out_prefix + "_data.bin")
    idmap_path = Path(out_prefix + "_idmap.bin")
    print("Writing to:", data_path, idmap_path)

    # open files in binary streaming mode
    data_f = open(data_path, 'wb')
    idmap_f = open(idmap_path, 'wb')

    # Pre-compile the struct for idmap records for a minor speedup
    idmap_struct = struct.Struct('<IIQ') # (term_id:uint32, node_id:uint32, row_index:uint64)

    # helper to write idmap record
    def write_idmap_record(term_id: int, node_id: int, row_index: int):
        idmap_f.write(idmap_struct.pack(term_id, node_id, row_index))

    # per-term loop
    row_index = 0
    num_terms = len(terms)

    for tidx, term in enumerate(tqdm(terms, desc="Processing Terms")):
        term_id = vocab[term.getTerm()]
        term = term.getTerm()
        postings = reader.postings(term)
        if not postings:
            continue

        # --- OPTIMIZATION: Get DF once per term ---
        df = reader.df(term)
        if df <= 0:
            continue
        # Pre-calculate IDF once per term
        tidf = idf(df, N)

        # build leaf_max array (max tf per leaf)
        num_leaves = layout.num_leaves
        leaf_max = np.zeros(num_leaves, dtype=np.int32)

        for docid, tf in postings:
            leaf_idx = docid // B
            if leaf_idx < num_leaves:
                if tf > leaf_max[leaf_idx]:
                    leaf_max[leaf_idx] = tf

        # if all zeros, skip
        if not leaf_max.any():
            continue

        # --- OPTIMIZATION: Vectorized bottom-up aggregation ---
        # Start with the max_tf values at the leaf level
        current_maxes = leaf_max
        h = layout.height

        # Loop from the parent-of-leaf level (h-1) up to the root (0)
        for level in range(h-1, -1, -1):
            nodes_in_level = layout.actual_nodes_at_level(level) # <-- *** THIS IS THE FIX ***
            node_id_start = layout.node_level_start(level)

            # Pad the current_maxes array so its length is a multiple of r
            # This handles partially filled levels at the fringe of the tree
            padded_len = int(np.ceil(current_maxes.shape[0] / r) * r)
            if padded_len > current_maxes.shape[0]:
                padded_maxes = np.zeros(padded_len, dtype=np.int32)
                padded_maxes[:current_maxes.shape[0]] = current_maxes
            else:
                padded_maxes = current_maxes

            # Reshape to (num_parents, r)
            # child_maxes_per_node[i, j] is the max_tf for j-th child of i-th node
            child_maxes_per_node = padded_maxes.reshape(-1, r)

            # We only care about the nodes that actually exist at this level
            if child_maxes_per_node.shape[0] > nodes_in_level:
                child_maxes_per_node = child_maxes_per_node[:nodes_in_level]

            # Compute parent maxes for the *next* level up (for the next iteration)
            # Shape: (nodes_in_level,)
            parent_maxes = child_maxes_per_node.max(axis=1)

            # Find which nodes at *this* level have any non-zero children
            # `any_nonzero_rows` is a boolean array of shape (nodes_in_level,)
            any_nonzero_rows = (child_maxes_per_node > 0).any(axis=1)

            # Get the indices of the nodes we need to write
            # node_indices_to_write is an array of node_idx values (e.g., [0, 2, 5, ...])
            node_indices_to_write = np.where(any_nonzero_rows)[0]

            # If there are no nodes with data at this level, skip to the next level
            if node_indices_to_write.size == 0:
                current_maxes = parent_maxes
                continue

            # This array will hold the r child scores for a *single* node
            child_scores = np.zeros(r, dtype=np.float32)

            # Now, iterate *only* over the nodes that have non-zero data
            for node_idx in node_indices_to_write:
                # Get the r max_tf values for the children of this node
                child_maxes = child_maxes_per_node[node_idx] # Shape: (r,)
                node_id = node_id_start + node_idx

                # Compute BM25 UB score for each of the r children
                for j in range(r):
                    mtf = int(child_maxes[j])
                    if mtf <= 0:
                        child_scores[j] = 0.0
                        continue

                    # Get min_dl for the child node
                    child_node_id = node_id * r + j
                    min_dl = layout.min_dl_map.get(child_node_id, int_avgdl)

                    # Calculate score
                    sc = bm25_term_ub(mtf, min_dl, avgdl, k1=k1, b=bparam)
                    child_scores[j] = sc * tidf # Apply pre-computed IDF

                # --- OPTIMIZATION: Vectorized scaling and quantization ---
                # Apply scaling: value -> int(round(value * scale)), clip 0..255
                q_scores = np.round(child_scores * scale)
                q_scores = np.clip(q_scores, 0, 255).astype(np.uint8)

                # Write the row to data.bin and idmap
                data_f.write(bytearray(q_scores))
                write_idmap_record(term_id, node_id, row_index)
                row_index += 1

            # Done with this level, move up
            current_maxes = parent_maxes

    data_f.close()
    idmap_f.close()
    print("Done. Wrote", row_index, "rows.")
    print("Files:")
    print(f"  {data_path} ({data_path.stat().st_size} bytes)")
    print(f"  {idmap_path} ({idmap_path.stat().st_size} bytes)")
    print(f"  {vocab_path}")


if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Build stage-1 PIR DB (data.bin + idmap.bin + vocab.json)")
    parser.add_argument('--index', required=True, help="Path to Lucene/Pyserini index")
    parser.add_argument('--vocab', required=False, help="Path to vocabulary file (one term per line). If omitted, set --terms-from-index")
    parser.add_argument('--terms-from-index', action='store_true', help="Enumerate terms from index (may be slow / pyserini dependent)")
    parser.add_argument('--layout-stats', required=False, help="Path to data/layout_with_stats.json (optional) to get min_dl_map")
    parser.add_argument('--out-prefix', default='stage1', help="Output prefix; files written: <prefix>_data.bin <prefix>_idmap.bin <prefix>_vocab.json")
    parser.add_argument('--B', type=int, default=32, help="Block size (default: 32)")
    parser.add_argument('--r', type=int, default=128, help="Fanout r (default: 128)")
    parser.add_argument('--scale', type=float, default=10.0, help="Scale factor before rounding to uint8 (default: 10.0)")
    parser.add_argument('--k1', type=float, default=0.9, help="BM25 k1 (default: 0.9)")
    parser.add_argument('--bparam', type=float, default=0.4, help="BM25 b (default: 0.4)")
    
    args = parser.parse_args()
    
    if args.r <= 1:
        parser.error("--r must be greater than 1")

    build(args)

