#!/usr/bin/env python3
"""
Plaintext, **cache‑free** simulation of the Privacy‑Preserving Sublinear Top‑k Retrieval Protocol
on MS MARCO (passage or document).

This version is optimized for higher recall by using **tighter upper bounds**
based on pre-computed block-level minimum document lengths (min_dl).

Key Optimizations Integrated:
  • **Tighter Bounds (Algorithmic Fix):** Instead of a loose, document-length-free
    upper bound, Rounds 1 & 2 now use a much more accurate bound derived from the
    pre-computed minimum document length (`min_dl`) within each block. This leads
    to more accurate pruning and higher recall.
  • **Decoupled Candidate Selection:** The number of candidate sub-blocks selected
    in Round 2 (`--k_candidates`) is now separate from the final retrieval size (`--k`),
    preventing the counter-intuitive drop in recall when increasing `k`.
  • **Centralized Query Data:** All per-query data (postings, IDFs) is fetched
    once and passed through the pipeline, avoiding redundant work.

Usage example:
  pip install pyserini==0.24.0 numpy tqdm
  python msmarco_plaintext_protocol_optimized.py \
    --index /path/to/lucene-index-msmarco-passage \
    --queries /path/to/queries.dev.small.tsv \
    --k 100 --k_candidates 1000 --L 400 \
    --r 128 --B 32 --s 8
"""

from __future__ import annotations
import argparse
import json
import math
import mmap  # Added
import struct  # Added
import os  # Added for path manipulation
from bisect import bisect_left
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from tqdm import tqdm
from collections import defaultdict
# ---- Pyserini glue ------------------------------------------------------------
try:
    from pyserini.index import LuceneIndexReader
    from pyserini.search.lucene import LuceneSearcher
except Exception as e:
    raise SystemExit("Pyserini is required: pip install pyserini\n" + str(e))


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

    _postings_cache: Dict[str, List[Tuple[int, int]]] = {}
    def postings(self, term: str) -> List[Tuple[int, int]]:
        if term in self._postings_cache:
            return self._postings_cache[term]
        pl = self.reader.get_postings_list(term, analyzer=None)
        lst = [] if pl is None else [(int(p.docid), int(p.tf)) for p in pl]
        self._postings_cache[term] = lst
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


    # def bm25_topk(self, query: str, k: int) -> List[Tuple[int, float]]:
    #     hits = self.searcher.search(query, k=k)
    #     out = []
    #     for h in hits:
    #         iid = self.reader.convert_collection_docid_to_internal_docid(h.docid)
    #         out.append((int(iid), float(h.score)))
    #     return out


# ---- BM25 & bounds ------------------------------------------------------------
@dataclass
class BM25Params:
    k1: float = 0.9
    b: float = 0.4

def idf(df: int, N: int) -> float:
    return math.log(1 + (N - df + 0.5) / (df + 0.5))

def bm25_term(tf: int, dl: int, avgdl: float, p: BM25Params) -> float:
    num = (p.k1 + 1.0) * tf
    denom = tf + p.k1 * (1.0 - p.b + p.b * (dl / max(1.0, avgdl)))
    return num / max(1e-9, denom)

# --- ALGORITHMIC FIX 1: TIGHTER BOUNDS ---
# This new scoring function uses the pre-computed minimum document length (min_dl)
# of a block to create a much tighter, more accurate upper bound.
def bm25_term_ub_with_min_dl(tf: int, min_dl: int, avgdl: float, p: BM25Params) -> float:
    return bm25_term(tf, min_dl, avgdl, p)

@dataclass
class QueryData:
    terms: List[str]
    docids: Dict[str, List[int]]
    tfs: Dict[str, List[int]]
    idfs: Dict[str, float]

    @classmethod
    def build(cls, terms: List[str], adp: LuceneAdapter) -> "QueryData":
        postings = {t: adp.postings(t) for t in terms}
        docids = {t: [d for d, _ in lst] for t, lst in postings.items()}
        tfs = {t: [tf for _, tf in lst] for t, lst in postings.items()}
        idfs = {t: idf(adp.df(t), adp.N) for t in terms}
        return cls(terms, docids, tfs, idfs)

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


# ---- Round 1: Beam search with Tighter Bounds -----------------------------
def round1_beam(layout: LayoutWithStats, query_data: QueryData, L: int, p: BM25Params, adp: LuceneAdapter) -> List[int]:
    frontier = [1] # Always start at root
    H = layout.height()

    for _ in range(H):
        children = layout.node_children(frontier)
        if not children or children == frontier: # Handle r=1 or leaf case
             break
        scores: List[Tuple[float, int]] = []
        for c in children:
            a, b = layout.node_to_doc_range(c)
            # Fetch the pre-computed min_dl for this block
            min_dl_for_block = layout.min_dl_map.get(c, adp.avgdl)
            if min_dl_for_block == float('inf'):
                min_dl_for_block = adp.avgdl
            
            total = 0.0
            for t in query_data.terms:
                dlist, tfs = query_data.docids[t], query_data.tfs[t]
                if not dlist: continue
                
                i = bisect_left(dlist, a)
                j = bisect_left(dlist, b, lo=i)
                if i < j:
                    mtf = max(tfs[i:j])
                    # Use the new, tighter scoring function
                    total += bm25_term_ub_with_min_dl(mtf, min_dl_for_block, adp.avgdl, p) * query_data.idfs[t]
            scores.append((total, c))
        scores.sort(key=lambda x: x[0], reverse=True)
        frontier = [c for _, c in scores[:L]]

    return frontier

# ---- Round 2: Sub-block selection with Tighter Bounds -----------------------
@dataclass
class HitSubBlock:
    start: int; end: int; score_ub: float

def round2_bmw(layout: LayoutWithStats, leaf_nodes: List[int], query_data: QueryData, s: int, k_candidates: int, p: BM25Params, adp: LuceneAdapter) -> List[HitSubBlock]:
    hits: List[HitSubBlock] = []
    avgdl = adp.avgdl

    for node in leaf_nodes:
        a, b = layout.node_to_doc_range(node)
        x = a
        while x < b:
            y = min(b, x + s)
            
            # Here, we could pre-compute min_dl for sub-blocks too for max accuracy.
            # For simplicity, we use the parent leaf's min_dl as a reasonable proxy.
            min_dl_for_block = layout.min_dl_map.get(node, avgdl)
            if min_dl_for_block == float('inf'):
                min_dl_for_block = adp.avgdl

            total = 0.0
            for t in query_data.terms:
                dlist, tfs = query_data.docids[t], query_data.tfs[t]
                if not dlist: continue

                i = bisect_left(dlist, x)
                j = bisect_left(dlist, y, lo=i)
                if i < j:
                    mtf = max(tfs[i:j])
                    total += bm25_term_ub_with_min_dl(mtf, min_dl_for_block, avgdl, p) * query_data.idfs[t]
            
            if total > 0: # Only append if score is non-zero
                hits.append(HitSubBlock(x, y, total))
            x = y
            
    hits.sort(key=lambda h: h.score_ub, reverse=True)
    return hits[:k_candidates]

import json
import mmap
import os
import struct
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Set
from tqdm import tqdm
# --- your existing dataclass ---
@dataclass
class HitSubBlock:
    start: int
    end: int
    score_ub: float


# -------- Stage-2 precomputed store loader (OPTIMIZED) --------
class Stage2Precomputed:
    """
    Reader for precomputed Round-2 bounds written by build_stage2_pir_db.py.

    OPTIMIZED: Builds a full nested lookup: term_id -> node_id -> row_index
    at startup for O(1) query-time lookups.
    """
    REC = struct.Struct("<IIQ")  # 16 bytes
    RECSZ = REC.size

    def __init__(self, data_path: str, idmap_path: str, vocab_path: str, B: int, s: int, scale: float):
        assert B % s == 0, "B must be divisible by s"
        self.data_path = data_path
        self.idmap_path = idmap_path
        self.vocab_path = vocab_path
        self.B = B
        self.s = s
        self.r = B // s
        self.scale = float(scale)

        # mmap files
        self._data_f = open(self.data_path, "rb")
        self._idmap_f = open(self.idmap_path, "rb")
        self.data = mmap.mmap(self._data_f.fileno(), 0, access=mmap.ACCESS_READ)
        self.idmap = mmap.mmap(self._idmap_f.fileno(), 0, access=mmap.ACCESS_READ)

        # vocab map: term -> term_id
        self.term_to_id: Dict[str, int] = {}
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                vocab_raw = json.load(f)
            self.term_to_id = {term: int(tid) for term, tid in vocab_raw.items()}
            print(f"Loaded Stage-2 vocab with {len(self.term_to_id)} terms.")
        except Exception as e:
            raise IOError(f"Error loading {self.vocab_path}: {e}")

        # Build full lookup table: term_id -> node_id -> row_index
        self.lookup_table: Dict[int, Dict[int, int]] = {}
        self._build_lookup_table()

    def close(self):
        try:
            self.data.close()
        finally:
            try:
                self.idmap.close()
            finally:
                self._data_f.close()
                self._idmap_f.close()

    def _num_records(self) -> int:
        return len(self.idmap) // self.RECSZ

    def _build_lookup_table(self):
        """Builds a fast, nested dictionary for O(1) lookups."""
        nrec = self._num_records()
        if nrec == 0:
            return
        
        print("Building Stage-2 lookup table...")
        for i in tqdm(range(nrec), desc="  - Loading Stage-2 idmap"):
            off = i * self.RECSZ
            term_id, node_id, row_index = self.REC.unpack_from(self.idmap, off)
            
            if term_id not in self.lookup_table:
                self.lookup_table[term_id] = {}
            self.lookup_table[term_id][node_id] = int(row_index)
        
        print(f"Built Stage-2 lookup table with {nrec} (term, node) entries.")

    def _rows_for_term_nodes(self, term_id: int, wanted_nodes: Set[int]) -> Dict[int, int]:
        """
        Return {node_id -> row_index} for the subset of leaf nodes we care about.
        Uses the pre-built lookup table for O(1) access per node.
        """
        out: Dict[int, int] = {}
        term_id = int(term_id)
        
        # Get the sub-map for this term
        term_map = self.lookup_table.get(term_id)
        if not term_map or not wanted_nodes:
            return out
        
        # O(k) operation where k = len(wanted_nodes)
        for node_id in wanted_nodes:
            row_index = term_map.get(node_id)
            if row_index is not None:
                out[node_id] = row_index
        return out

    def _read_row_bytes(self, row_index: int) -> memoryview:
        byte_off = row_index * self.r
        return memoryview(self.data)[byte_off: byte_off + self.r]

    def sum_bounds_for_terms(
        self,
        terms: Iterable[str],
        leaf_nodes: Iterable[int],
    ) -> Dict[int, np.ndarray]:
        """
        For the given terms and leaf_nodes, return:
          leaf_node -> np.ndarray shape (r,), dtype=int32
        representing the SUM across terms of the precomputed per-subblock bounds (in byte domain).
        """
        wanted_nodes: Set[int] = set(leaf_nodes)
        if not wanted_nodes:
            return {}

        # map terms -> ids; ignore OOV terms
        term_ids = [self.term_to_id[t] for t in terms if t in self.term_to_id]
        if not term_ids:
            # no terms exist in precomputation
            return {ln: np.zeros(self.r, dtype=np.int32) for ln in wanted_nodes}

        # accumulators per leaf
        acc: Dict[int, np.ndarray] = {ln: np.zeros(self.r, dtype=np.int32) for ln in wanted_nodes}

        # per term, find rows for the wanted leaf nodes, then add
        for tid in term_ids:
            rows = self._rows_for_term_nodes(tid, wanted_nodes)
            if not rows:
                continue
            for ln, row_idx in rows.items():
                row_mv = self._read_row_bytes(row_idx)
                # convert to vector without copying then cast up to avoid uint8 wrap
                vec = np.frombuffer(row_mv, dtype=np.uint8).astype(np.int32, copy=False)
                acc[ln] += vec

        return acc

# ---- NEW: Stage-1 PIR Database Loader -----------------------------------------
class Stage1PIRDB:
    """
    Loads the pre-computed Stage-1 DB files (vocab, idmap, data).
    Provides a fast lookup table: (term_id, node_id) -> row_index
    and an mmap-ed data file for score lookups.
    """
    def __init__(self, data_path: str, idmap_path: str, vocab_path: str, r: int):
        self.r = r
        self.data_mmap = None
        self.data_file = None
        self.lookup_table: Dict[int, Dict[int, int]] = {} # term_id -> node_id -> row_index
        self.vocab: Dict[str, int] = {} # term_str -> term_id

        print("Loading Stage-1 PIR Database...")
        
        # 1. Load Vocab (term_str -> term_id)
        # Assumes vocab IDs are integers (or string ints)
        try:
            vocab = json.load(open(vocab_path, 'r', encoding='utf-8'))
            self.vocab = {t: int(tid) for t, tid in vocab.items()} # convert to int term id
            print(f"  - Loaded vocab with {len(self.vocab)} terms.")
        except Exception as e:
            raise IOError(f"Error loading {vocab_path}: {e}")

        # 2. Load idmap and build lookup table
        try:
            # Struct format: < (little-endian), I (uint32), I (uint32), Q (uint64)
            record_format = struct.Struct('<IIQ')
            record_size = record_format.size # 4 + 4 + 8 = 16 bytes
            
            with open(idmap_path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                f.seek(0, 0)
                
                if file_size % record_size != 0:
                    print(f"Warning: {idmap_path} size ({file_size}) is not a multiple of record size ({record_size})")

                num_records = file_size // record_size
                
                for _ in tqdm(range(num_records), desc="  - Loading idmap"):
                    buf = f.read(record_size)
                    if len(buf) < record_size: break
                    
                    term_id, node_id, row_index = record_format.unpack(buf)
                    
                    if term_id not in self.lookup_table:
                        self.lookup_table[term_id] = {}
                    self.lookup_table[term_id][node_id] = int(row_index)
            print(f"  - Built lookup table with {num_records} (term, node) entries.")

        except Exception as e:
            raise IOError(f"Error loading {idmap_path}: {e}")

        # 3. Open data.bin via mmap
        try:
            self.data_file = open(data_path, 'rb')
            self.data_mmap = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Sanity check
            expected_rows = num_records
            expected_size = expected_rows * self.r
            if len(self.data_mmap) < expected_size:
                 print(f"Warning: {data_path} size ({len(self.data_mmap)}) is smaller than expected ({expected_size})")
            print(f"  - Mapped {data_path} ({len(self.data_mmap)} bytes).")

        except Exception as e:
            self.close() # Clean up
            raise IOError(f"Error mapping {data_path}: {e}")

    def close(self):
        """Closes the mmap-ed file."""
        if self.data_mmap:
            self.data_mmap.close()
            self.data_mmap = None
        if self.data_file:
            self.data_file.close()
            self.data_file = None



# ---- NEW: Round 1: Beam search (PRE-COMPUTED lookup) -----------------------
def round1_beam_precomputed(
    layout: LayoutWithStats, 
    query_terms: List[str], 
    L: int, 
    db: Stage1PIRDB
) -> List[int]:
    """
    Performs Round 1 beam search using the pre-computed Stage-1 database.
    Scores are uint8, so we accumulate them as integers.
    """
    frontier = [1] # Always start at root
    H = layout.height()
    r = db.r
    
    # 1. Convert query term strings to term_ids
    query_term_ids: List[int] = []
    for term in query_terms:
        term_id = db.vocab.get(term)
        if term_id is not None:
            query_term_ids.append(term_id)
            
    if not query_term_ids:
        return []

    # 2. Loop through each level of the tree
    for level in range(H):
        # child_scores maps: child_node_id -> accumulated_score (sum of uint8s)
        child_scores = defaultdict(int)

        # 3. For each parent node in the current beam
        for parent_node_id in frontier:
            # 4. For each term in the query
            for term_id in query_term_ids:
                
                # 5. Look up the row_index for this (term, parent_node)
                term_map = db.lookup_table.get(term_id)
                if term_map is None:
                    continue # This term has no data
                
                row_index = term_map.get(parent_node_id)
                if row_index is None:
                    continue # This (term, node) has no scores (all children are 0)

                # 6. Fetch the score row (r bytes) from mmap
                try:
                    start_offset = row_index * r
                    score_row = db.data_mmap[start_offset : start_offset + r]
                except Exception as e:
                    print(f"Error reading mmap at index {row_index} (offset {start_offset}): {e}")
                    continue
                
                # 7. Add scores to all 'r' children
                for j in range(r):
                    # Child node ID logic must match build script
                    child_node_id = parent_node_id * r + j
                    
                    # score_row[j] is already a uint8 int
                    child_scores[child_node_id] += score_row[j]

        if not child_scores:
            # print(f"  - Level {level}: Search ended early (no children found).")
            break # No children had any scores

        # 8. Prune: Sort all scored children by their total score
        # Note: We sort by value (score) descending
        sorted_children = sorted(child_scores.items(), key=lambda item: item[1], reverse=True)
        
        # 9. Set new frontier (top L node_ids)
        frontier = [node_id for node_id, score in sorted_children[:L]]

        # print(f"  - Level {level}: Frontier size {len(frontier)}, Top score: {sorted_children[0][1]}")

    return frontier


def round2_bmw_precomputed(
    layout,
    leaf_nodes: List[int],
    query_terms: List[str],
    s: int,
    k_candidates: int,
    pre: Stage2Precomputed,
) -> List[HitSubBlock]:
    """
    Round 2 using precomputed Stage-2 files.

    - Sums per-subblock bounds across query terms for each leaf.
    - Emits top-k_candidates subblocks globally.
    """
    assert pre.s == s, f"Precomputed s={pre.s} does not match requested s={s}"
    r = pre.r
    
    # 1) Sum per-subblock UBs in BYTE domain for each leaf
    sums_by_leaf = pre.sum_bounds_for_terms(query_terms, leaf_nodes)

    hits: List[HitSubBlock] = []

    # 2) Expand each leaf into at most r sub-blocks and attach summed scores
    for node in leaf_nodes:
        a, b = layout.node_to_doc_range(node)
        if a >= b:
            continue
        vec = sums_by_leaf.get(node)
        if vec is None:
            # If no term had rows for this node, all subblocks are zero → skip
            continue

        # Create sub-blocks; last leaf may be partial, so stop when start >= b
        for j in range(r):
            start = a + j * s
            if start >= b:
                break
            end = min(b, start + s)

            # Score: divide by scale to get back to approx float UB (optional)
            score_ub = float(vec[j]) / pre.scale

            if score_ub <= 0.0:
                # you can keep zeroes if you prefer stable tie-breaking
                continue

            hits.append(HitSubBlock(start=start, end=end, score_ub=score_ub))

    # 3) Global top-k by score
    hits.sort(key=lambda h: h.score_ub, reverse=True)
    if k_candidates is not None and k_candidates > 0:
        hits = hits[:k_candidates]
    return hits


# ---- Round 3: Exact Scoring (Optimized Accumulator Pattern) ---------------
def round3_score_exact(adp: LuceneAdapter, query_data: QueryData, hits: List[HitSubBlock], p: BM25Params, k: int) -> List[Tuple[int,float]]:
    candidate_dids = {d for h in hits for d in range(h.start, h.end)}
    if not candidate_dids: return []

    doc_lengths = {did: adp.doc_len(did) for did in candidate_dids}
    scores = {did: 0.0 for did in candidate_dids}
    avgdl = adp.avgdl

    for t in query_data.terms:
        term_idf = query_data.idfs[t]
        if term_idf <= 0: continue

        # We need the full postings list here, not just docids/tfs
        postings = adp.postings(t)
        
        # This could be faster if we start scan from first potential candidate
        for did, tf in postings:
            if did in scores:
                dl = doc_lengths.get(did)
                if dl is not None and dl > 0:
                    scores[did] += bm25_term(tf, dl, avgdl, p) * term_idf
    
    scored_docs = [(did, score) for did, score in scores.items() if score > 0]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:k]

# ---- Experiment & Main ------------------------------------------------------
def parse_tsv_queries(path: str, max_q: Optional[int]) -> List[Tuple[str,str]]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_q is not None and i >= max_q: break
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 2: out.append((parts[0], parts[1]))
    return out

def load_qrels(qrels_path: str) -> Dict[str, Set[str]]:
    """Loads a qrels file into a dictionary for fast lookups."""
    qrels = {}
    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            qid, _, docid, _ = line.strip().split()
            if qid not in qrels:
                qrels[qid] = set()
            qrels[qid].add(docid)
    return qrels

def experiment(args):
    adp = LuceneAdapter(args.index)
    p = BM25Params(k1=args.k1, b=args.bparam)
    adp.searcher.set_bm25(k1=args.k1, b=args.bparam)

    qs = parse_tsv_queries(args.queries, args.max_queries)
    # qs = qs[974+1520+1445:]  # continue from last stopped point
    qrels = load_qrels(args.qrels)
    
    # --- FIX: Calculate r_round2 from B and s ---
    if args.B % args.s != 0:
        raise SystemExit(f"Error: B ({args.B}) must be divisible by s ({args.s})")
    r_round2 = args.B // args.s
    if r_round2 <= 0:
         raise SystemExit(f"Error: r_round2 (B/s) must be > 0, but got {r_round2}. Check B and s.")
    print(f"Round 1 arity r = {args.r}")
    print(f"Round 2 sub-block arity r_round2 = B/s = {args.B}/{args.s} = {r_round2}")

    stage1_db = Stage1PIRDB(args.stage1_data_bin, args.stage1_idmap_bin, args.vocab, args.r)

    pir_db = None
    if args.use_pir_db:
        if not all([args.data_bin, args.idmap_bin, args.vocab]):
            raise SystemExit("Error: --use-Pir-db requires --data-bin, --idmap-bin, and --vocab")
        # Pass r_round2 to the DB loader
        pir_db = Stage2PIRDB(args.data_bin, args.idmap_bin, args.vocab, r_round2)
    
    if not args.precompute:    
        # --- SETUP ---
        # Create the layout and run the one-time preprocessing step for min_dl stats.
        # Pass r_round1 (args.r) to the Layout
        print(f"Building layout with N={adp.N}, B={args.B}, r={args.r}")
        layout = LayoutWithStats(N=adp.N, B=args.B, r=args.r)
        layout.precompute_stats(adp)

        # save layout to storage
        with open('data/layout_with_stats.json', 'w') as f:
            json.dump({
                'N': layout.N,
                'B': layout.B,
                'r': layout.r, # This is r_round1
                'min_dl_map': {k: v for k, v in layout.min_dl_map.items() if v != float('inf')} # Don't save inf
            }, f, indent=2)
    else:
        #load from storage:
        print("Loading precomputed layout from data/layout_with_stats.json...")
        with open('data/layout_with_stats.json', 'r') as f:
            data = json.load(f)
        
        # --- FIX: Check for consistency ---
        if data['B'] != args.B or data['r'] != args.r:
            print(f"Warning: Layout file parameters (B={data['B']}, r={data['r']}) "
                  f"do not match command line (B={args.B}, r={args.r}).")
            print("This may cause incorrect results. Recommend deleting data/layout_with_stats.json and re-running.")
            print(f"Layout file r={data['r']}, args.r={args.r}")

        layout = LayoutWithStats(N=data['N'], B=data['B'], r=data['r'])
        # Convert string keys back to int
        layout.min_dl_map = {int(k): v for k,v in data.get('min_dl_map', {}).items()}
        print("Layout loaded.")

    pre = Stage2Precomputed(
                    data_path='stage2/data.bin',
                    idmap_path='stage2/idmap.bin',
                    vocab_path='stage1_vocab.json',
                    B=args.B,
                    s=args.s,
                    scale=10.0
                )
    recalls = []
    recalls_bm25 = []
    for qid, qtext in tqdm(qs, desc='Queries'):
        t = time.time()
        terms = adp.analyze(qtext)
        if not terms: continue

        # bm25 = adp.bm25_topk(qtext, args.k)
        baseline = qrels.get(qid, set())
        #map to internal docids
        # gold = {adp.reader.convert_collection_docid_to_internal_docid(did) for did in baseline}

        gold = {did for did in baseline}
        if not gold: continue

        # query_data = QueryData.build(terms, adp)
        # print("Query processing time:", time.time() - t)
        t = time.time()
        # --- PROTOCOL ---
        # Round 1 uses layout with r_round1 (args.r)
        # leaf_nodes = round1_beam(layout, query_data, L=args.L, p=p, adp=adp) # online computation
        leaf_nodes = round1_beam_precomputed(
                layout, 
                terms, # Pass raw analyzed terms
                L=args.L, 
                db=stage1_db
            )
        # print(f"Round 1 time: {time.time() - t:.2f}s for qid {qid} with {len(leaf_nodes)} leaf nodes")
        t = time.time()
        # --- MODIFIED: Choose Round 2 implementation ---
        if args.use_pir_db:
            # Round 2 uses pir_db (with r_round2)
            hit_subs = round2_pir_db(
                layout, leaf_nodes, query_data, 
                k_candidates=args.k_candidates, 
                pir_db=pir_db, 
                scale=args.scale
            )
        else:
            hit_subs = round2_bmw_precomputed(
                layout, leaf_nodes, terms, 
                s=args.s, 
                k_candidates=args.k_candidates, 
                pre=pre
            )
            
            # hit_subs = round2_bmw(
            #     layout, leaf_nodes, query_data, 
            #     s=args.s, 
            #     k_candidates=args.k_candidates, 
            #     p=p, 
            #     adp=adp
            # )
            # s1,s2= set(),set()
            # for h1,h2 in zip(hit_subs1, hit_subs):
            #     s1.add((h1.start,h1.end))
            #     s2.add((h2.start,h2.end))
            # print("Intersection R2 precomputed vs online:", len(s1.intersection(s2)), " / ", len(s1), len(s2))
            # assert False
        # print(f"Round 2 time: {time.time() - t:.2f}s for qid {qid} with {len(hit_subs)} hit sub-blocks")
        approx = {d for h in hit_subs for d in range(h.start, h.end)}
        approx = [(d,1.0) for d in approx]  # dummy score for compatibility
        # approx = round3_score_exact(adp, query_data, hit_subs, p=p, k=args.k)
        # approx_ids = {did for did, _ in approx}
        
        if not approx:
            # print(f"Warning: No documents found for qid {qid}")
            pass

        approx_ids = set()
        for did, score in approx:
            try:
                ext_did = adp.reader.convert_internal_docid_to_collection_docid(did)
                approx_ids.add(ext_did)
            except Exception as e:
                # This can happen if docid is out of bounds
                # print(f"Error converting docid {did} for qid {qid}: {e}")
                pass # skip this docid

        #write approx to file along with qid to file
        with open(f'approx_precompute_{args.k}_{args.L}_{args.k_candidates}.tsv', 'a') as f:
            for ext_did in approx_ids:
                f.write(f"{qid}\t{ext_did}\t{1.0}\n") # Using dummy score 1.0
        
        # inter = len(gold & approx_ids)
        # rec = inter / len(gold)
        # recalls.append(rec)

        # compute recall for bm25 baseline too

        # bm25_ids = {did for did, _ in bm25}
        # print(gold, approx_ids, bm25_ids)

        # inter_bm25 = len(gold & bm25_ids)
        # rec_bm25 = inter_bm25 / len(gold)
        # recalls_bm25.append(rec_bm25)

    if recalls:
        arr = np.array(recalls)
        print(f"\n--- Results (k={args.k}, L={args.L}, k_candidates={args.k_candidates}) ---")
        print(f"Evaluated {len(recalls)} queries")
        print(f"Recall@{args.k}: mean={arr.mean():.4f} median={np.median(arr):.4f} p90={np.quantile(arr, 0.9):.4f}")
    else:
        print("No queries with relevant documents were evaluated.")
    # if recalls_bm25:
    #     arr_bm25 = np.array(recalls_bm25)
    #     print(f"BM25 Recall@{args.k}: mean={arr_bm25.mean():.4f} median={np.median(arr_bm25):.4f} p90={np.quantile(arr_bm25, 0.9):.4f}")
    # else:
    #     print("No BM25 recalls computed.")

    # Clean up the mmap
    if pir_db:
        pir_db.close()

def main():
    ap = argparse.ArgumentParser(description='Optimized Cache‑free PP top‑k simulation on MS MARCO')
    ap.add_argument('--index', required=True, help="Path to Pyserini index")
    ap.add_argument('--queries', required=True, help="Path to queries.tsv file")
    ap.add_argument('--B', type=int, default=32, help="Block size")
    # --- RE-ADDED r as an argument ---
    ap.add_argument('--r', type=int, default=128, help="Arity of the conceptual tree for Round 1")
    ap.add_argument('--L', type=int, default=200, help="Beam width for Round 1")
    ap.add_argument('--s', type=int, default=8, help="Sub-block size for Round 2")
    ap.add_argument('--k', type=int, default=100, help="Final number of documents to retrieve for recall@k")
    ap.add_argument('--k_candidates', type=int, default=1000, help="Number of candidate sub-blocks to select in Round 2")
    ap.add_argument('--max-queries', type=int, default=200, help="Maximum number of queries to run")
    ap.add_argument('--k1', type=float, default=0.9, help="BM25 k1 parameter")
    ap.add_argument('--b', dest='bparam', type=float, default=0.4, help="BM25 b parameter")
    ap.add_argument('--qrels', required=True, help="Path to qrels file for evaluation")
    ap.add_argument('--precompute', action='store_true', help="If set, load precomputed layout and stats from storage instead of recomputing")

    # --- NEW ARGS ---
    ap.add_argument('--use-Pir-db', action='store_true',
                      help="Use pre-computed Stage-2 PIR database for Round 2")
    ap.add_argument('--data-bin', type=str, default=None, 
                      help="Path to data.bin (required if --use-Pir-db)")
    ap.add_argument('--idmap-bin', type=str, default=None, 
                      help="Path to idmap.bin (required if --use-Pir-db)")
    ap.add_argument('--vocab', type=str, default=None, 
                      help="Path to vocab.txt (required if --use-Pir-db)")
    ap.add_argument('--scale', type=float, default=10.0, 
                      help="Scale factor (must match DB build scale)")
    ap.add_argument('--stage1-data-bin', type=str, default=None, 
                      help="Path to Stage-1 data.bin (required)")
    ap.add_argument('--stage1-idmap-bin', type=str, default=None, 
                      help="Path to Stage-1 idmap.bin (required)")
    
    args = ap.parse_args()
    
    experiment(args)

if __name__ == '__main__':
    main()
