"""
Loads every run under RESULTS_DIR into

    nested_data[method][dataset] = [run, run, ...]

where each run is a flat dict (keys listed under RUN KEYS below). Run
`python load_results.py` for a summary, or
`python load_results.py <folder>` to print one run.

FOLDERS
-------
One folder per run, e.g. /scratch/dblackle/results/bins_vec0_msmarco_k10_bs100_dpb10/
containing metadata.json (the only file read here), results.json,
results_reRank.json, slurm_name.txt. Folders are created before the run
finishes, so a folder without metadata.json is skipped (counted in the log).

    bins_vec{0|1}_{dataset}_k{K}_bs{BS}_dpb{DPB}
        vec0     two DBs: BM25 bins DB, then a separate embedding (Vec) DB  -> "2-stage"
        vec1     one single DB holding everything                           -> "1-stage"
        dataset  msmarco | scifact
        k        documents returned per query in the final list (10 or 100)
        bs       bin size: number of entries in the DB (same as RealBinSize)
        dpb      docs per bin: documents returned by one entry, so the DB
                 stores bs * dpb documents in total

    pacmann_{dataset}_k{K}_steps{S}_neighb{N}
        steps    rounds of approximate nearest neighbour search before stopping
        neighb   neighbours uncovered per round (used to pick the next hop)

    tree_{dataset}_..._k{K}
        Layout not fixed yet, see parse_tree().

METADATA.JSON (all values are strings)
-------------
Common to every method:
    ClientStorageMB     size of the client's PIR hint
    CommCostPerBatchKB  KB sent for one batched PIR query
    DBSizeInBytesMB     server DB size (MB, despite the name)
    FailureProbLog2     PIR failure probability, log2
    MRR / Recall        of the final list output by the method
    MRRPreReRank / RecallPreReRank
                        before the local rerank (cosine similarity against the
                        query embedding, keep top-k)
    NumQueries          queries in the run, used to make everything per-query
    PreprocessingTime   PIR preprocessing, Go duration ('11.17s'), one-off
    ReRankTime          local rerank time, Go duration, summed over all queries
    TotalAnswerTime     raw computation for all queries, Go duration ('5m36.1s')
    TotalByteSent       bytes up + down over the whole run
    TotalLANTime        simulated LAN time for the whole run, seconds
    TotalWANTime        simulated WAN time for the whole run, seconds
    Faithfulness        (added later) does the LLM stick to the retrieved context
    AnswerRelevancy     (added later) usefulness of the answer itself
    Pre*                values from before the run, ignored

Bins only:
    EmptyBins           entries holding no documents
    RealBinSize         entries in the DB (should equal bs in the folder name)
    VocabSize           unique words in the corpus
    Vec*                same meaning as above, but for the embedding DB. Only
                        present for vec0. Totals are main + Vec, e.g. the full
                        LAN time is TotalLANTime + VecTotalLANTime. There is no
                        VecTotalAnswerTime.

Pacmann only:
    GraphGenerationTime graph building, a pacmann-only preprocessing step

RUN KEYS
--------
Per query (divided by NumQueries):
    total_time   TotalAnswerTime (s)
    rerank_time  ReRankTime (s)
    lan_time     TotalLANTime (+ VecTotalLANTime) (s)
    wan_time     TotalWANTime (+ VecTotalWANTime) (s)
    comm_kb      TotalByteSent (+ VecTotalByteSent) in KB
Not per query:
    mrr, mrr_pre_rerank, recall, recall_pre_rerank, faithfulness,
    answer_relevancy, db_size_mb, client_storage_mb, comm_per_batch_kb,
    failure_prob_log2, preprocessing_time (s, one-off),
    graph_generation_time (s, pacmann only), empty_bins, vocab_size,
    real_bin_size (bins only)
Split keys (bins only): lan_time, wan_time, comm_kb, db_size_mb,
client_storage_mb and comm_per_batch_kb also exist as <key>_bm25 (main DB)
and <key>_vec (embedding DB). The unsuffixed key is always the total. For
vec1, <key>_vec is 0.
Labels: folder, method, dataset, k, config, num_queries, and per method
vec/bs/dpb (bins) or steps/neighb (pacmann).
Anything missing from metadata.json is NaN, and summarised at the end.
"""

import json
import logging
import math
import os
import re
import sys
from collections import Counter

import globals as g

log = logging.getLogger('load')

BINS_NAME = re.compile(r'^bins_vec(?P<vec>\d)_(?P<dataset>[a-z0-9-]+)_k(?P<k>\d+)_bs(?P<bs>\d+)_dpb(?P<dpb>\d+)$')
PACMANN_NAME = re.compile(r'^pacmann_(?P<dataset>[a-z0-9-]+)_k(?P<k>\d+)_steps(?P<steps>\d+)_neighb(?P<neighb>\d+)$')
TREE_NAME = re.compile(r'^tree_(?P<dataset>[a-z0-9-]+)_(?P<config>.+)$')

DURATION_PART = re.compile(r'(\d+(?:\.\d+)?)(h|ms|µs|μs|us|ns|m|s)')
DURATION_UNITS = {'h': 3600, 'm': 60, 's': 1, 'ms': 1e-3, 'µs': 1e-6, 'μs': 1e-6, 'us': 1e-6, 'ns': 1e-9}


# ==========================================
# VALUE READERS
# ==========================================

def read_float(meta, key):
    """meta[key] as a float, NaN if missing or unreadable."""
    if key not in meta:
        return math.nan
    try:
        return float(meta[key])
    except (TypeError, ValueError):
        log.warning("could not read %s=%r as a number", key, meta[key])
        return math.nan


def read_seconds(meta, key):
    """Seconds from either a plain number ('34.91') or a Go duration ('5m36.15s', '101ms')."""
    if key not in meta:
        return math.nan
    value = str(meta[key]).strip()
    try:
        return float(value)
    except ValueError:
        pass
    parts = DURATION_PART.findall(value)
    if not parts:
        log.warning("could not read %s=%r as a duration", key, value)
        return math.nan
    return sum(float(num) * DURATION_UNITS[unit] for num, unit in parts)


def read_db_part(meta, n, prefix=''):
    """
    Values reported once per DB. prefix='' reads the main DB, prefix='Vec'
    the embedding DB of a bins vec0 run.
    """
    return {
        'lan_time':          read_seconds(meta, prefix + 'TotalLANTime') / n,
        'wan_time':          read_seconds(meta, prefix + 'TotalWANTime') / n,
        'comm_kb':           read_float(meta, prefix + 'TotalByteSent') / 1024 / n,
        'db_size_mb':        read_float(meta, prefix + 'DBSizeInBytesMB'),
        'client_storage_mb': read_float(meta, prefix + 'ClientStorageMB'),
        'comm_per_batch_kb': read_float(meta, prefix + 'CommCostPerBatchKB'),
    }


# ==========================================
# PARSERS (one per method)
# ==========================================

def parse_common(folder, meta):
    """Keys every method shares. Returns None if the run can't be made per-query."""
    n = read_float(meta, 'NumQueries')
    if not n > 0:
        log.error("%s: NumQueries is %r, skipping (can't compute per-query values)",
                  folder, meta.get('NumQueries'))
        return None

    run = {
        'folder':             folder,
        'num_queries':        int(n),
        'mrr':                read_float(meta, 'MRR'),
        'mrr_pre_rerank':     read_float(meta, 'MRRPreReRank'),
        'recall':             read_float(meta, 'Recall'),
        'recall_pre_rerank':  read_float(meta, 'RecallPreReRank'),
        'faithfulness':       read_float(meta, 'Faithfulness'),
        'answer_relevancy':   read_float(meta, 'AnswerRelevancy'),
        'total_time':         read_seconds(meta, 'TotalAnswerTime') / n,
        'rerank_time':        read_seconds(meta, 'ReRankTime') / n,
        'preprocessing_time': read_seconds(meta, 'PreprocessingTime'),   # one-off, not per query
        'failure_prob_log2':  read_float(meta, 'FailureProbLog2'),
    }
    run.update(read_db_part(meta, n))

    if run['mrr'] == 0:
        log.warning("%s: MRR is 0", folder)
    if run['comm_kb'] == 0:
        log.warning("%s: TotalByteSent is 0", folder)
    return run


def parse_bins(folder, meta):
    name = BINS_NAME.match(folder)
    if not name:
        log.warning("%s: doesn't match bins_vec<0|1>_<dataset>_k<K>_bs<BS>_dpb<DPB>, skipping", folder)
        return None
    run = parse_common(folder, meta)
    if run is None:
        return None

    vec, bs, dpb = int(name['vec']), int(name['bs']), int(name['dpb'])
    run.update(method='bins', dataset=name['dataset'], k=int(name['k']),
               vec=vec, bs=bs, dpb=dpb, config=f'vec{vec}_bs{bs}_dpb{dpb}',
               real_bin_size=read_float(meta, 'RealBinSize'),
               empty_bins=read_float(meta, 'EmptyBins'),
               vocab_size=read_float(meta, 'VocabSize'))

    has_vec_keys = 'VecTotalByteSent' in meta
    if vec == 0 and not has_vec_keys:
        log.warning("%s: vec0 run without Vec* keys, totals will be NaN", folder)
    if vec == 1 and has_vec_keys:
        log.warning("%s: vec1 run has Vec* keys, they are ignored", folder)
    if run['real_bin_size'] != bs:
        log.warning("%s: RealBinSize=%s but folder says bs=%d", folder, meta.get('RealBinSize'), bs)

    # Split into main (BM25) DB and embedding DB; the plain key is the total.
    main = read_db_part(meta, run['num_queries'])
    if vec == 0:
        second = read_db_part(meta, run['num_queries'], prefix='Vec')
    else:
        second = {key: 0.0 for key in main}
    for key in main:
        run[key + '_bm25'] = main[key]
        run[key + '_vec'] = second[key]
        run[key] = main[key] + second[key]
    return run


def parse_pacmann(folder, meta):
    name = PACMANN_NAME.match(folder)
    if not name:
        log.warning("%s: doesn't match pacmann_<dataset>_k<K>_steps<S>_neighb<N>, skipping", folder)
        return None
    run = parse_common(folder, meta)
    if run is None:
        return None

    steps, neighb = int(name['steps']), int(name['neighb'])
    run.update(method='pacmann', dataset=name['dataset'], k=int(name['k']),
               steps=steps, neighb=neighb, config=f'steps{steps}_neighb{neighb}',
               graph_generation_time=read_seconds(meta, 'GraphGenerationTime'))   # one-off
    return run


def parse_tree(folder, meta):
    """
    Placeholder until the tree folder layout and metadata keys are fixed.
    Reads the common keys, the dataset and k; everything else in the folder
    name is kept as `config`.
    """
    name = TREE_NAME.match(folder)
    k = re.search(r'_k(\d+)', folder)
    if not (name and k):
        log.warning("%s: can't find dataset and _k<K> in tree folder name, skipping", folder)
        return None
    run = parse_common(folder, meta)
    if run is None:
        return None

    config = re.sub(r'_?k\d+$', '', name['config']).strip('_') or 'default'
    run.update(method='tree', dataset=name['dataset'], k=int(k.group(1)), config=config)
    # TODO: tree-specific keys go here
    return run


PARSERS = {'bins': parse_bins, 'pacmann': parse_pacmann, 'tree': parse_tree}


# ==========================================
# LOADING
# ==========================================

def load_results(results_dir=g.RESULTS_DIR):
    nested_data = {method: {} for method in g.METHOD_ORDER}
    if not os.path.isdir(results_dir):
        log.error("results directory does not exist: %s", results_dir)
        return nested_data

    no_metadata, bad_json, skipped = [], [], []
    for folder in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, folder)
        if not os.path.isdir(path):
            continue

        method = folder.split('_', 1)[0]
        if method not in PARSERS:
            log.warning("%s: unknown method, skipping", folder)
            continue

        meta_path = os.path.join(path, 'metadata.json')
        if not os.path.exists(meta_path):
            no_metadata.append(folder)
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except json.JSONDecodeError as err:
            log.error("%s: corrupted metadata.json (%s)", folder, err)
            bad_json.append(folder)
            continue

        run = PARSERS[method](folder, meta)
        if run is None:
            skipped.append(folder)
            continue
        log.debug("loaded %s", folder)
        nested_data.setdefault(method, {}).setdefault(run['dataset'], []).append(run)

    if no_metadata:
        log.warning("%d folders have no metadata.json yet, e.g. %s",
                    len(no_metadata), ', '.join(no_metadata[:4]))
    if bad_json or skipped:
        log.warning("%d corrupted, %d skipped: %s", len(bad_json), len(skipped),
                    ', '.join(bad_json + skipped))
    print_summary(nested_data)
    return nested_data


def print_summary(nested_data):
    """One line per method/dataset: run count, k values, and keys that are NaN."""
    for method, per_dataset in nested_data.items():
        if not per_dataset:
            log.warning("%s: no runs loaded", method)
        for dataset, runs in sorted(per_dataset.items()):
            ks = sorted({r['k'] for r in runs})
            nan_counts = Counter(key for r in runs for key, value in r.items()
                                 if isinstance(value, float) and math.isnan(value))
            missing = ', '.join(f'{key} ({n})' for key, n in nan_counts.most_common())
            log.info("%-8s %-9s %4d runs, k=%s%s", method, dataset, len(runs), ks,
                     f" | NaN: {missing}" if missing else "")


def find_run(nested_data, folder):
    """The run loaded from `folder`, or None. Handy for filling in hand-picked values."""
    for per_dataset in nested_data.values():
        for runs in per_dataset.values():
            for run in runs:
                if run['folder'] == folder:
                    return run
    return None


if __name__ == '__main__':
    g.setup_logging()
    data = load_results(g.RESULTS_DIR)
    for name in sys.argv[1:]:
        found = find_run(data, name)
        if found is None:
            print(f"{name}: not loaded")
            continue
        print(f"\n{name}")
        for key, value in found.items():
            print(f"  {key:<24} {value}")
