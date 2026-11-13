"""
Loads every run under g.RESULTS_DIRS (RESULTS_DIR, ':'-separated), falling back to
g.SIMULATED_RESULTS_DIRS for configs that have no real run (see REAL VS SIMULATED), into

    nested_data[method][dataset] = [run, run, ...]

where each run is a flat dict (keys listed under RUN KEYS below). Run
`python load_results.py` for a summary, or
`python load_results.py <folder>` to print one run.

REAL VS SIMULATED
-----------------
With g.FALLBACK_TO_SIMULATED (the default; FALLBACK_TO_SIMULATED=0 turns it off, or pass
fallback_simulated=False), each config is taken from the first place that has a
metadata.json for it: RESULTS_DIRS, the extra baseline dirs, then SIMULATED_RESULTS_DIRS.
run['source'] says which ('real' / 'simulated'). With g.ONLY_SIMULATED (ONLY_SIMULATED=1, or pass
only_simulated=True) only SIMULATED_RESULTS_DIRS and the Tiptoe/Coeus dirs (EXTRA_RESULTS_DIRS, always simulated)
are read: no real runs, no fallback. Simulated runs write -1 for every server
timing, so any negative value is loaded as NaN (total_time, preprocessing_time, ...);
quality, bytes, LAN/WAN and client storage are the same as a real run's. Sub-folders
named like h3_lf0.88/ (msmarco-v2's tree runs with other Cuckoo settings) are read too;
their runs' folder is 'h3_lf0.88/<name>'.

CLIENT STORAGE FILTER
---------------------
With g.FILTER_CLIENT_STORAGE (default on; FILTER_CLIENT_STORAGE=0, or pass
filter_client=False), runs of every method whose client_storage_mb is
>= g.MAX_CLIENT_FRACTION (1.0) x g.EMBED_DB_MB[dataset] (the dataset's embedding file)
are dropped after loading, unless that would drop every run of a method on a dataset (then
that method keeps all its runs there). The count per method/dataset is logged and the dropped runs
are kept in DROPPED[method][dataset]. Datasets missing from EMBED_DB_MB keep all their
runs (with a warning).

FOLDERS
-------
One folder per run, e.g. /scratch/dblackle/results/bins_vec0_msmarco_k10_bs100_dpb10/
containing metadata.json (the only file read here), results.json,
results_reRank.json, slurm_name.txt. Folders are created before the run
finishes, so a folder without metadata.json is skipped (counted in the log).

    bins_vec{0|1}_{dataset}_k{K}_bs{BS}_dpb{DPB}[_vb{B}_vr{R}_ib{I}]
        _vb_vr_ib  vec0 client-storage sweep: embedding-DB batch, rounds, ID-DB batch (g.INCLUDE_VEC0_SWEEP;
                 run['vec_sweep'] is True for these, and pu.bins_sweep leaves them out of the bs/dpb ablations)
        vec0     two DBs: BM25 bins DB, then a separate embedding (Vec) DB  -> "2-stage"
        vec1     one single DB holding everything                           -> "1-stage"
    bins_text_{dataset}_k{K}_bs{BS}_dpb{DPB}
        text     one DB of compressed doc text, client BM25 re-rank (-text)  -> vec=2,
                 loaded as more 'bins' runs (vec=2, config text_bs<BS>_dpb<DPB>)
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

    tiptoe_{dataset}_c{C}[_m{M}][_b{B}][_h{H}]_k{K}
        C        number of k-means clusters (baselines/tiptoe/preprocess_tiptoe.py)
        M        clusters each doc is assigned to (default 2)
        B        bundle size (default 160)
        H        embedding hint MB (default 500)

    coeus_{dataset}_k{K}_f{F}[_tf{raw|log}]
        F        keyword dictionary size (baselines/coeus/run_coeus.py)
        tf       term-frequency weighting (default raw)

METADATA.JSON (all values are strings)
-------------
Common to every method:
    ClientStorageMB     size of the client's PIR hint
    CommCostPerBatchKB  KB sent for one batched PIR query
    DBSizeInBytesMB     server DB size (MB, despite the name)
    FailureProbLog2     PIR failure probability, log2
    MRR / Recall        of the final list output by the method. For the
                        methods in g.MRR_FROM_PRE_RERANK (Coeus, Tiptoe) the
                        loaded 'mrr' is MRRPreReRank instead, see parse_common()
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

Tiptoe only (Tiptoe* keys, see tiptoe/tiptoe.go):
    TiptoeTokenKBPerQuery   offline per-query token (counted in TotalByteSent,
                            not in LAN/WAN time, like PIR preprocessing)
    TiptoeOfflineLANTime / TiptoeOfflineWANTime   network time of those tokens

Coeus only (Coeus* keys, see baselines/coeus/run_coeus.py):
    TotalAnswerTime is only present when Coeus' opt2 benchmark was calibrated.

RUN KEYS
--------
Per query (divided by NumQueries):
    total_time   TotalAnswerTime (s)
    rerank_time  ReRankTime (s)
    lan_time     TotalLANTime (+ VecTotalLANTime) (s)
    wan_time     TotalWANTime (+ VecTotalWANTime) (s)
    comm_kb      TotalByteSent (+ VecTotalByteSent) in KB
Not per query:
    mrr          MRR, or MRRPreReRank for g.MRR_FROM_PRE_RERANK methods
    mrr_reranked MRR as written in metadata.json, whatever the method
    mrr_pre_rerank, recall, recall_pre_rerank, faithfulness,
    answer_relevancy, db_size_mb, client_storage_mb, comm_per_batch_kb,
    failure_prob_log2, preprocessing_time (s, one-off),
    graph_generation_time (s, pacmann only), empty_bins, vocab_size,
    real_bin_size (bins only)
    pir_rounds   sequential PIR rounds per query: bins 1 (vec1) or 2 (vec0),
                 pacmann = steps, tree = ceil(sum of Stage{N}Rounds / NumQueries)
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
from concurrent.futures import ThreadPoolExecutor

import globals as g

log = logging.getLogger('load')

# bins_text_* = the -text mode (compressed doc text in one DB, client BM25 re-rank); parsed as vec=2
# optional _vb<B>_vr<R>_ib<I>: the vec0 client-storage sweep (run_bins_*.slurm add_vec0_sweep: embedding-DB batch,
# rounds, ID-DB batch, round-robin candidates)
BINS_NAME = re.compile(r'^bins_(?:vec(?P<vec>\d)|(?P<text>text))_(?P<dataset>[a-z0-9-]+)_k(?P<k>10)_bs(?P<bs>\d+)_dpb(?P<dpb>\d+)'
                       r'(?:_vb(?P<vb>\d+)_vr(?P<vr>\d+)_ib(?P<ib>\d+))?$')
PACMANN_NAME = re.compile(r'^pacmann_(?P<dataset>[a-z0-9-]+)_k(?P<k>10)_steps(?P<steps>\d+)_neighb(?P<neighb>\d+)$')
TREE_NAME = re.compile(r'^tree_(?:(?P<grid>grid)_)?(?P<dataset>[a-z0-9-]+)_b(?P<b>\d+)_r(?P<r>\d+)_s(?P<s>\d+)_L(?P<L>\d+)(?:_q(?P<q>\d+))?(?:_c(?P<c>\d+))?_k(?P<k>10)$')

CUCKOO_SUBDIR = re.compile(r'^h\d+_lf[0-9.]+$')   # tree results with non-default Cuckoo settings

TREE_STAGES = [1, 2, 3]   # each stage is one PIR DB; totals are the sum over these

DURATION_PART = re.compile(r'(\d+(?:\.\d+)?)(h|ms|µs|μs|us|ns|m|s)')
DURATION_UNITS = {'h': 3600, 'm': 60, 's': 1, 'ms': 1e-3, 'µs': 1e-6, 'μs': 1e-6, 'us': 1e-6, 'ns': 1e-9}


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
        'maintenance_time': read_seconds(meta, prefix + 'MaintainenceTime'),  # all preproc calls
        'preprocessing_time': read_seconds(meta, prefix + 'PreprocessingTime'),  # first call only
        'preproc_rounds': read_float(meta, prefix + 'TotalPreProc'),  # number of calls
    }

def combine_parts(run, parts):
    """
    Writes one set of per-DB values per part plus their total. `parts` is
    {suffix: {key: value}}, e.g. {'s1': {...}, 's2': {...}, 's3': {...}}.
    run[key] ends up as the sum, run[key_suffix] as that part's own value.
    """
    for key in next(iter(parts.values())):
        for suffix, values in parts.items():
            run[f'{key}_{suffix}'] = values[key]
        run[key] = sum(values[key] for values in parts.values())



def parse_common(folder, meta, method):
    """Keys every method shares. Returns None if the run can't be made per-query."""
    n = read_float(meta, 'NumQueries')
    if not n > 0:
        log.error("%s: NumQueries is %r, skipping (can't compute per-query values)",
                  folder, meta.get('NumQueries'))
        return None

    mrr_key = 'MRRPreReRank' if method in g.MRR_FROM_PRE_RERANK else 'MRR'
    run = {
        'folder':             folder,
        'num_queries':        int(n),
        'mrr':                read_float(meta, mrr_key),
        'mrr_reranked':       read_float(meta, 'MRR'),
        'mrr_pre_rerank':     read_float(meta, 'MRRPreReRank'),
        'recall':             read_float(meta, 'Recall'),
        'recall_pre_rerank':  read_float(meta, 'RecallPreReRank'),
        # rag-eval/ragas_eval.py writes Faithfulness / AnswerRelevancy; older runs used the lowercase keys
        'faithfulness':       read_float(meta, 'Faithfulness' if 'Faithfulness' in meta else 'faithfulness'),
        'answer_relevancy':   read_float(meta, 'AnswerRelevancy' if 'AnswerRelevancy' in meta else 'answer_relevancy'),
        'total_time':         read_seconds(meta, 'TotalAnswerTime') / n,
        'rerank_time':        read_seconds(meta, 'ReRankTime') / n,
        'preprocessing_time': read_seconds(meta, 'PreprocessingTime'),   # one-off, not per query
        'failure_prob_log2':  read_float(meta, 'FailureProbLog2'),
    }
    run.update(read_db_part(meta, n))

    if run['mrr'] == 0:
        log.warning("%s: %s is 0", folder, mrr_key)
    if run['comm_kb'] == 0:
        log.warning("%s: TotalByteSent is 0", folder)
    return run


def parse_bins(folder, meta):
    name = BINS_NAME.match(folder)
    if not name:
        if not "k100" in folder:
            log.warning("%s: doesn't match bins_vec<0|1>_<dataset>_k<K>_bs<BS>_dpb<DPB>, skipping", folder)
        return None
    run = parse_common(folder, meta, 'bins')
    if run is None:
        return None

    vec = 2 if name['text'] else int(name['vec'])
    bs, dpb = int(name['bs']), int(name['dpb'])
    tag = 'text' if vec == 2 else f'vec{vec}'
    sweep = name['vb'] is not None
    if sweep and not g.INCLUDE_VEC0_SWEEP:
        return None
    sweep_tag = f"_vb{name['vb']}_vr{name['vr']}_ib{name['ib']}" if sweep else ''
    # text runs (vec=2) are PILLAR-Bin configs like the others (config 'text_bs.._dpb..'); the bs/dpb ablations
    # still keep them apart, since they always ask for vec=0 or vec=1
    run.update(method='bins', dataset=name['dataset'], k=int(name['k']),
               vec=vec, bs=bs, dpb=dpb, config=f'{tag}_bs{bs}_dpb{dpb}{sweep_tag}',
               # vec0 sweep: the batching asked for (VecBatchSize in metadata is what was used: pir_app cuts a batch
               # too big for the embedding DB) and how many candidates per query batch x rounds had no room for
               vec_sweep=sweep, vec_batch=read_float(meta, 'VecBatchSize'), vec_rounds=read_float(meta, 'VecRounds'),
               id_batch=read_float(meta, 'IDBatchSize'), vec_dropped=read_float(meta, 'VecDroppedPerQuery'),
               vec_candidates=read_float(meta, 'VecCandidatesPerQuery'),
               pir_rounds=2 if vec == 0 else 1,   # vec0 queries the BM25 DB, then the Vec DB
               real_bin_size=read_float(meta, 'RealBinSize'),
               empty_bins=read_float(meta, 'EmptyBins'),
               vocab_size=read_float(meta, 'VocabSize'))

    has_vec_keys = 'VecTotalByteSent' in meta
    if vec == 0 and not has_vec_keys:
        log.warning("%s: vec0 run without Vec* keys, totals will be NaN", folder)
    if vec >= 1 and has_vec_keys:
        log.warning("%s: single-DB run has Vec* keys, they are ignored", folder)
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

    # est = estimate_client(meta, bs, vec, dpb)
    # run['client_storage_mb_reported'] = run['client_storage_mb']
    # run['client_storage_mb_bm25'] = est['client_mb_best']
    # run['client_storage_mb_low'] = est['client_mb_low'] + run['client_storage_mb_vec']
    # run['client_storage_mb'] = est['client_mb_low'] + run['client_storage_mb_vec']
    # if not est['consistent']:
    #     log.warning("%s: ClientStorageMB/CommCostPerBatchKB don't fit bs=%d, estimate is suspect", folder, bs)
    return run


def parse_pacmann(folder, meta):
    name = PACMANN_NAME.match(folder)
    if not name:
        if not "k100" in folder:
            log.warning("%s: doesn't match pacmann_<dataset>_k<K>_steps<S>_neighb<N>, skipping", folder)
        return None
    run = parse_common(folder, meta, 'pacmann')
    if run is None:
        return None

    steps, neighb = int(name['steps']), int(name['neighb'])
    run.update(method='pacmann', dataset=name['dataset'], k=int(name['k']),
               steps=steps, neighb=neighb, config=f'steps{steps}_neighb{neighb}',
               graph_generation_time=read_seconds(meta, 'GraphGenerationTime'),   # one-off
               pir_rounds=steps)   # one PIR round per graph hop
    return run


def parse_tree(folder, meta):
    """
    Three PIR stages, each with its own DB and its own Stage{N}* keys. The
    end-to-end value is the sum over the stages, and each stage is also kept as
    <key>_s1 / _s2 / _s3 for the stacked latency plot. A missing stage counts
    as 0 (stage 3 is still buggy) and is logged once per run.
    """
    name = TREE_NAME.match(folder)
    if not name:
        if not "k100" in folder:
            log.warning("%s: doesn't match tree_<dataset>_b<B>_r<R>_s<S>_L<L>[_q<Q>][_c<C>]_k<K>, skipping", folder)
        return None
    run = parse_common(folder, meta, 'tree')
    if run is None:
        return None

    b, r, s, L = int(name['b']), int(name['r']), int(name['s']), int(name['L'])
    # q / c (k_candidates) only appear in newer folder names; keep them in the label so
    # configs that differ only in q or c don't collapse into one point.
    q = int(name['q']) if name['q'] else None
    c = int(name['c']) if name['c'] else None
    config = f'b{b}_r{r}_s{s}_L{L}' + (f'_q{q}' if q is not None else '') + (f'_c{c}' if c is not None else '')
    run.update(method='tree', dataset=name['dataset'], k=int(name['k']),
               b=b, r=r, s=s, L=L, q=q, k_candidates=c, config=config)

    parts, missing = {}, []
    for stage in TREE_STAGES:
        if f'Stage{stage}TotalLANTime' in meta:
            parts[f's{stage}'] = read_db_part(meta, run['num_queries'], prefix=f'Stage{stage}')
        else:
            missing.append(stage)
    if not parts:
        log.error("%s: no Stage* keys at all, skipping", folder)
        return None
    if missing:
        log.warning("%s: stage(s) %s missing, counted as 0 in the totals",
                    folder, ', '.join(map(str, missing)))
        zero = {key: 0.0 for key in next(iter(parts.values()))}
        parts.update({f's{stage}': dict(zero) for stage in missing})

    combine_parts(run, {f's{stage}': parts[f's{stage}'] for stage in TREE_STAGES})
    run['pir_rounds'] = tree_pir_rounds(folder, meta, run['num_queries'])
    if run['num_queries'] == 300:
        run['maintenance_time'] = run['maintenance_time'] * 3.9
    return run


def tree_pir_rounds(folder, meta, n):
    """
    Stage{N}Rounds are totals over the whole run, so the per-query count is
    their sum / NumQueries, rounded up (a query can't do part of a round).
    NaN if no stage reports its rounds.
    """
    keys = [f'Stage{stage}Rounds' for stage in TREE_STAGES if f'Stage{stage}Rounds' in meta]
    if not keys:
        log.warning("%s: no Stage*Rounds keys, pir_rounds is NaN", folder)
        return math.nan
    total = sum(read_float(meta, key) for key in keys)
    return math.nan if math.isnan(total) else math.ceil(total / n)


# m = multi-assign, b = bundle size, h = embedding hint MB (optional, older runs don't have them)
TIPTOE_NAME = re.compile(r'^tiptoe_(?P<dataset>[a-z0-9-]+)_c(?P<c>\d+)(?:_m(?P<m>\d+))?(?:_b(?P<b>\d+))?(?:_h(?P<h>\d+))?_k(?P<k>10)$')
COEUS_NAME = re.compile(r'^coeus_(?P<dataset>[a-z0-9-]+)_k(?P<k>10)_f(?P<f>\d+)(?:_tf(?P<tf>raw|log))?$')


def parse_tiptoe(folder, meta):
    name = TIPTOE_NAME.match(folder)
    if not name:
        if "k100" not in folder:
            log.warning("%s: doesn't match tiptoe_<dataset>_c<C>_k<K>, skipping", folder)
        return None
    run = parse_common(folder, meta, 'tiptoe')
    if run is None:
        return None
    c = int(name['c'])
    m, b, h = int(name['m'] or 2), int(name['b'] or 160), int(name['h'] or 500)
    run.update(method='tiptoe', dataset=name['dataset'], k=int(name['k']), c=c, multi_assign=m, bundle=b,
               emb_hint_mb=h, config=f'c{c}_m{m}_b{b}_h{h}',
               pir_rounds=2,   # ranking round + URL round (the token is fetched offline)
               token_kb=read_float(meta, 'TiptoeTokenKBPerQuery'),
               offline_wan_time=read_seconds(meta, 'TiptoeOfflineWANTime') / run['num_queries'])
    return run


def parse_coeus(folder, meta):
    name = COEUS_NAME.match(folder)
    if not name:
        if "k100" not in folder:
            log.warning("%s: doesn't match coeus_<dataset>_k<K>_f<F>, skipping", folder)
        return None
    run = parse_common(folder, meta, 'coeus')
    if run is None:
        return None
    f, tf = int(name['f']), name['tf'] or 'raw'
    run.update(method='coeus', dataset=name['dataset'], k=int(name['k']), f=f, tf=tf, config=f'f{f}_tf{tf}',
               pir_rounds=2)   # encrypted scoring, then metadata PIR
    return run


PARSERS = {'bins': parse_bins, 'pacmann': parse_pacmann, 'tree': parse_tree,
           'tiptoe': parse_tiptoe, 'coeus': parse_coeus}


# ==========================================
# LOADING
# ==========================================

def _load_folder(folder, candidates):
    """
    Reads and parses one run folder. `folder` is relative to a results dir (maybe 'h3_lf0.88/<name>'),
    `candidates` the [(results_dir, source), ...] that have it, best first: the first with a metadata.json wins.
    Returns (status, method_or_None, run_or_None) where status is 'ok', 'not_dir', 'unknown_method',
    'no_metadata', 'bad_json', or 'skipped'. Runs in a worker thread, so it must not touch shared state.
    """
    name = os.path.basename(folder)
    method = name.split('_', 1)[0]
    if method not in PARSERS:
        return ('unknown_method' if any(os.path.isdir(os.path.join(d, folder)) for d, _ in candidates)
                else 'not_dir'), None, None

    for results_dir, source in candidates:
        meta_path = os.path.join(results_dir, folder, 'metadata.json')
        if os.path.exists(meta_path):
            break
    else:
        return 'no_metadata', None, None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except json.JSONDecodeError as err:
        log.error("%s (%s): corrupted metadata.json (%s)", folder, results_dir, err)
        return 'bad_json', None, None

    run = PARSERS[method](name, meta)
    if run is None:
        return 'skipped', None, None
    # simulated runs write -1 for every timing they don't measure: NaN, not a negative per-query time
    for key, value in run.items():
        if isinstance(value, float) and value < 0:
            run[key] = math.nan
    run.update(folder=folder, source=source, results_dir=results_dir)
    return 'ok', run.get('method', method), run


def _list_folders(results_dir):
    """Run folders in results_dir, relative to it: the top level plus h<H>_lf<LF>/ sub-folders' contents."""
    out = []
    for name in sorted(os.listdir(results_dir)):
        if CUCKOO_SUBDIR.match(name) and os.path.isdir(os.path.join(results_dir, name)):
            out += [os.path.join(name, sub) for sub in sorted(os.listdir(os.path.join(results_dir, name)))]
        else:
            out.append(name)
    return out


def load_results(results_dirs=None, max_workers=32, simulated_dirs=None, fallback_simulated=None,
                 filter_client=None, only_simulated=None):
    """
    Reads every folder's metadata.json in `results_dirs` (one path, a list of
    paths, or None for g.RESULTS_DIRS) with a thread pool. With fallback_simulated
    (default g.FALLBACK_TO_SIMULATED), a config with no metadata.json there is read
    from `simulated_dirs` (default g.SIMULATED_RESULTS_DIRS) instead; run['source']
    says which. Loading is I/O-bound (many small file reads, often over a network
    filesystem), so threads give a big speedup despite the GIL: it's released during
    file I/O, and json.load / the parsers only hold it for the small compute part.
    filter_client (default g.FILTER_CLIENT_STORAGE) drops the runs (every method) whose
    client storage reaches the embedding DB size, see filter_client_storage().
    only_simulated (default g.ONLY_SIMULATED) reads `simulated_dirs` and nothing else:
    results_dirs is ignored (the Tiptoe/Coeus EXTRA_RESULTS_DIRS are still read), and a config with no simulated
    run is simply missing (it never falls back to a real run).
    """
    if results_dirs is None:
        results_dirs = g.RESULTS_DIRS
    elif isinstance(results_dirs, str):
        results_dirs = [d for d in results_dirs.split(os.pathsep) if d]
    if fallback_simulated is None:
        fallback_simulated = g.FALLBACK_TO_SIMULATED
    if simulated_dirs is None:
        simulated_dirs = g.SIMULATED_RESULTS_DIRS
    elif isinstance(simulated_dirs, str):
        simulated_dirs = [d for d in simulated_dirs.split(os.pathsep) if d]
    if only_simulated is None:
        only_simulated = g.ONLY_SIMULATED
    # Tiptoe / Coeus (EXTRA_RESULTS_DIRS) are always simulated runs, so they're read in both modes
    extra = [d for d in g.EXTRA_RESULTS_DIRS if d not in results_dirs and d not in simulated_dirs]
    for d in extra:
        if not os.path.isdir(d):
            log.debug("extra results directory not found, skipped: %s", d)
    extra_sources = [(d, 'simulated') for d in extra if os.path.isdir(d)]
    if only_simulated:
        if not simulated_dirs:
            raise ValueError("only_simulated is on but no simulated results dirs are set (SIMULATED_RESULTS_DIR)")
        sources = [(d, 'simulated') for d in simulated_dirs] + extra_sources
    else:
        sources = [(d, 'real') for d in results_dirs] + extra_sources
        if fallback_simulated:
            sources += [(d, 'simulated') for d in simulated_dirs if d not in results_dirs]
    log.info("reading results from %s", ', '.join(f'{d} ({src})' for d, src in sources))
    nested_data = {method: {} for method in g.METHOD_ORDER}

    # folder -> [(results_dir, source), ...] in priority order; each folder is loaded once, from the first of these
    # with a metadata.json (a real run beats a simulated one; a real folder without metadata.json falls through)
    candidates = {}
    for results_dir, source in sources:
        if not os.path.isdir(results_dir):
            (log.warning if source == 'simulated' else log.error)("results directory does not exist: %s", results_dir)
            continue
        is_extra = (results_dir, source) in extra_sources
        for folder in _list_folders(results_dir):
            # a mixed extra folder (datasets/results) only gives its baselines, see g.EXTRA_METHODS
            if is_extra and g.EXTRA_METHODS and os.path.basename(folder).split('_', 1)[0] not in g.EXTRA_METHODS:
                continue
            candidates.setdefault(folder, []).append((results_dir, source))
    jobs = sorted(candidates.items())
    no_metadata, bad_json, skipped = [], [], []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # pool.map yields results in `jobs` order even though workers finish
        # out of order, so nested_data ends up built deterministically.
        for (folder, _), (status, method, run) in zip(
                jobs, pool.map(lambda job: _load_folder(*job), jobs)):
            if status == 'not_dir':
                continue
            if status == 'unknown_method':
                log.warning("%s: unknown method, skipping", folder)
                continue
            if status == 'no_metadata':
                no_metadata.append(folder)
                continue
            if status == 'bad_json':
                bad_json.append(folder)
                continue
            if status == 'skipped':
                skipped.append(folder)
                continue
            log.debug("loaded %s", folder)
            if method == 'bins-text' and method not in g.METHOD_ORDER:   # g.TEXT_BINS = []
                continue
            nested_data.setdefault(method, {}).setdefault(run['dataset'], []).append(run)

    if no_metadata:
        log.warning("%d folders have no metadata.json yet, e.g. %s",
                    len(no_metadata), ', '.join(no_metadata[:4]))
    if bad_json or skipped:
        log.warning("%d corrupted, %d skipped: %s", len(bad_json), len(skipped),
                    ', '.join(bad_json + skipped))
    if g.FILTER_CLIENT_STORAGE if filter_client is None else filter_client:
        filter_client_storage(nested_data)
    exclude_client_storage(nested_data)
    print_summary(nested_data)
    return nested_data


def exclude_client_storage(nested_data, methods=None):
    """client_storage_mb -> NaN for g.STORAGE_EXCLUDE_METHODS (the reported value is kept as client_storage_mb_reported),
    so client-storage plots and tables leave those methods out while every other metric keeps them."""
    methods = g.STORAGE_EXCLUDE_METHODS if methods is None else methods
    for method in methods:
        n = 0
        for runs in nested_data.get(method, {}).values():
            for r in runs:
                r['client_storage_mb_reported'] = r.get('client_storage_mb', math.nan)
                r['client_storage_mb'] = math.nan
                n += 1
        if n:
            log.info("%s: client storage left out of the comparisons (%d runs; STORAGE_EXCLUDE_METHODS=\"\" keeps it)",
                     method, n)


DROPPED = {}   # method -> dataset -> runs the client storage filter took out (so reports can say what it hid)
DROPPED_BINS = DROPPED.setdefault('bins', {})   # same dict as DROPPED['bins'], for the older reports


def filter_client_storage(nested_data, fraction=None):
    """Drops runs of every method with client_storage_mb >= fraction x the dataset's embedding DB (g.EMBED_DB_MB),
    in place. Runs before exclude_client_storage(), so the reported value is what is compared. The dropped runs are
    kept in DROPPED[method][dataset]."""
    fraction = g.MAX_CLIENT_FRACTION if fraction is None else fraction
    for method, per_dataset in nested_data.items():
        for dataset, runs in per_dataset.items():
            limit = g.EMBED_DB_MB.get(dataset)
            if limit is None:
                log.warning("%s/%s: no embedding DB size in g.EMBED_DB_MB, client storage filter not applied",
                            method, dataset)
                continue
            limit *= fraction
            # NaN client storage compares False, so those runs are kept
            keep = [r for r in runs if not r.get('client_storage_mb', math.nan) >= limit]
            if runs and not keep:   # the filter would wipe out this method on this dataset: keep all of its runs
                log.info("%s/%s: all %d runs have client storage >= %.1f MB (%.2f x the %.1f MB embedding DB), "
                         "filter not applied so the method stays in", method, dataset, len(runs), limit, fraction,
                         g.EMBED_DB_MB[dataset])
                DROPPED.setdefault(method, {})[dataset] = []
                continue
            DROPPED.setdefault(method, {})[dataset] = [r for r in runs if r.get('client_storage_mb', math.nan) >= limit]
            if len(keep) < len(runs):
                log.info("%s/%s: dropped %d of %d runs with client storage >= %.1f MB (%.2f x the %.1f MB embedding DB)",
                         method, dataset, len(runs) - len(keep), len(runs), limit, fraction, g.EMBED_DB_MB[dataset])
            runs[:] = keep


def print_summary(nested_data):
    """One line per method/dataset: run count (real / simulated), k values, and keys that are NaN."""
    for method, per_dataset in nested_data.items():
        if not per_dataset:
            log.warning("%s: no runs loaded", method)
        for dataset, runs in sorted(per_dataset.items()):
            ks = sorted({r['k'] for r in runs})
            src = Counter(r.get('source', 'real') for r in runs)
            nan_counts = Counter(key for r in runs for key, value in r.items()
                                 if isinstance(value, float) and math.isnan(value))
            missing = ', '.join(f'{key} ({n})' for key, n in nan_counts.most_common(8))
            log.info("%-8s %-15s %4d runs (%d real, %d simulated), k=%s%s", method, dataset, len(runs),
                     src['real'], src['simulated'], ks, f" | NaN: {missing}" if missing else "")


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
    data = load_results()
    for name in sys.argv[1:]:
        found = find_run(data, name)
        if found is None:
            print(f"{name}: not loaded")
            continue
        print(f"\n{name}")
        for key, value in found.items():
            print(f"  {key:<24} {value}")
