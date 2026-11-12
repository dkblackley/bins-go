#!/bin/bash -l
#SBATCH --job-name=DocFetch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/DocFetch_%j.out
#SBATCH --error=logs/DocFetch_%j.err
#SBATCH --signal=B:TERM@120
#
# Document fetch (cmd/docfetch): the cost of privately downloading the k documents a method returns. One Piano PIR
# DB per dataset, one row per doc (compressed text), and every query fetches k docs in one batch (k = 10 and 100).
# All methods end with k doc IDs, so this one DB serves them all.
#
# Run it with bash (not sbatch): it builds the binary once, then submits one job per dataset, each sized for it.
#   bash run_docfetch.slurm                       # every dataset below, simulated PIR
#   bash run_docfetch.slurm nq fever              # only these
#   REAL=1 bash run_docfetch.slurm scifact msmarco nq   # real PIR: hints, server timings (more RAM, see MAXROW_KB)
#   DRY_RUN=1 bash run_docfetch.slurm             # print the sbatch lines only
#
# Results: $RESULTS_DIR/docfetch_<ds>_k<K>_flate/{metadata.json,fetched.json,sample_docs.json}
#   RESULTS_DIR default /scratch/$USER/results_docfetch (real) or .../results_docfetch_simulated (simulated)
# Knobs: DATASETS_ROOT (../datasets), K (10,100), COMPRESS (flate|none), MAX_DOC_BYTES (0 = never truncate),
#   IDS_DIR (a folder with <ds>.json = a method's results.json: fetch its answers instead of random docs),
#   MAX_QUERIES (0 = all), MEM / TIME (override the estimate for every job), MAXROW_KB (real mode: guessed largest
#   compressed doc, which sizes the client hint tables; 16), EXTRA_ARGS (more docfetch flags), FORCE=1 (rerun).
#
# Simulated runs give the exact bytes, LAN/WAN times and client storage (timings -1); REAL=1 adds server time. The
# client's hint tables hold every hint padded to the largest row, so on msmarco-v2* REAL=1 can need a lot of RAM
# unless MAX_DOC_BYTES caps the docs (the truncated count is in metadata.json).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
mkdir -p logs bin

DATASETS_ROOT="${DATASETS_ROOT:-../datasets}"
REAL="${REAL:-0}"
if [[ "$REAL" == 1 ]]; then
    RESULTS_DIR="${RESULTS_DIR:-/scratch/${USER:-$(id -un)}/results_docfetch}"
else
    RESULTS_DIR="${RESULTS_DIR:-/scratch/${USER:-$(id -un)}/results_docfetch_simulated}"
fi

# ------------------------------------------------------------------ inside a job: run one dataset
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    : "${DATASET:?set DATASET}" "${DOCFETCH_BIN:?set DOCFETCH_BIN}"
    export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-16}"
    args=(-name "$DATASET" -dataset "$DATASETS_ROOT" -out "$RESULTS_DIR" -k "${K:-10,100}"
          -compress "${COMPRESS:-flate}" -max-doc-bytes "${MAX_DOC_BYTES:-0}" -max-queries "${MAX_QUERIES:-0}")
    [[ "$REAL" == 1 ]] || args+=(-simulate)
    [[ "${FORCE:-0}" == 1 ]] && args+=(-force)
    [[ -n "${IDS_DIR:-}" ]] && args+=(-ids "$IDS_DIR/$DATASET.json")
    echo "docfetch $DATASET on $(hostname), $(date +%FT%T): $DOCFETCH_BIN ${args[*]} ${EXTRA_ARGS:-}"
    mkdir -p "$RESULTS_DIR"
    # peak RAM + wall time of the job, next to the results (for sizing reruns)
    # shellcheck disable=SC2086
    bash measure_run.sh "$RESULTS_DIR/docfetch_${DATASET}_time_v.txt" "$DOCFETCH_BIN" "${args[@]}" ${EXTRA_ARGS:-}
    exit $?
fi

# ------------------------------------------------------------------ on the login node: build, then one job per dataset
datasets="${*:-scifact msmarco nq fever dbpedia-entity msmarco-v2-25m msmarco-v2-50m msmarco-v2}"

# docs : words per doc (slurm_common.sh DATASET_SIZES); num_docs.txt wins when it exists
SIZES="msmarco:8841823:56 scifact:5183:214 trec-covid:171332:160 nq:2681468:79 dbpedia-entity:4635922:50
fever:5416568:85 hotpotqa:5233329:46 miracl-en:32893221:100 msmarco-v2:138364198:70 msmarco-v2-25m:25000000:70
msmarco-v2-50m:50000000:70 msmarco-v2-100m:100000000:70"

# estimate <docs> <words/doc> -> "<GB> <hours>"
#   rows ~ half the text (deflate) + ~150 B/doc of Go overhead (row + ID headers, the PIR's shuffle tables), x2.2 for
#   the GC; real mode adds the hint tables of the k = 100 run (50 partitions, every hint padded to MAXROW_KB).
estimate() {
    awk -v n="$1" -v w="$2" -v real="$REAL" -v maxrow="${MAXROW_KB:-16}" 'BEGIN {
        text = 6.5 * w + 40
        gb = n * (0.5 * text + 150) * 2.2 / 1e9 + 8
        if (real == 1) {
            parts = 50; p = n / parts; c = 1; while (c < 2 * sqrt(p)) c *= 2
            q = sqrt(p) * log(p)
            entries = parts * (15 * c + 6 * q)
            gb += entries * maxrow * 1024 * 1.3 / 1e9
        }
        gb = int(gb + 0.999); if (gb < 16) gb = 16
        h = 1 + n / 20e6 * (real == 1 ? 3 : 1); if (h > 48) h = 48
        printf "%d %d\n", gb, int(h + 0.999)
    }'
}

if [[ -z "${DRY_RUN:-}" ]]; then
    bin="bin/docfetch.$(date +%Y%m%d_%H%M%S)"   # own file per build: jobs still running keep theirs (NFS, see run_me.sh)
    ( source ./go_env.sh >/dev/null && go build -o "$bin" ./cmd/docfetch ) || { echo "build failed"; exit 1; }
    echo "built $bin"
else
    bin="bin/docfetch.DRYRUN"
fi
echo "mode: $([[ "$REAL" == 1 ]] && echo real || echo simulated) PIR, results -> $RESULTS_DIR"

for d in $datasets; do
    n=0 w=70
    for e in $SIZES; do [[ "${e%%:*}" == "$d" ]] && { IFS=: read -r _ n w <<< "$e"; break; }; done
    [[ -s "$DATASETS_ROOT/$d/num_docs.txt" ]] && n=$(<"$DATASETS_ROOT/$d/num_docs.txt")
    corpus="$DATASETS_ROOT/$d/corpus.jsonl"
    if [[ ! -s "$corpus" ]]; then echo "SKIP $d: no $corpus"; continue; fi
    if [[ "$n" == 0 ]]; then n=$(wc -l < "$corpus"); fi
    read -r gb hours <<< "$(estimate "$n" "$w")"
    gb="${MEM:-$gb}"; gb="${gb%G}"
    part=normal; ((gb > 100)) && part=bigmem
    time="${TIME:-$(printf '%02d:00:00' "$hours")}"
    cmd=(sbatch --parsable --job-name="DocFetch_$d" --partition="$part" --mem="${gb}G" --time="$time"
         --export="ALL,DATASET=$d,DOCFETCH_BIN=$bin,RESULTS_DIR=$RESULTS_DIR,DATASETS_ROOT=$DATASETS_ROOT,REAL=$REAL"
         run_docfetch.slurm)
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "$d ($n docs): ${cmd[*]}"
    else
        id=$("${cmd[@]}") && echo "$d ($n docs): job $id, ${gb}G $part, $time"
    fi
done
