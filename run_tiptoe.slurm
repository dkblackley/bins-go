#!/bin/bash -l
#SBATCH --job-name=Tiptoe_Baseline
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00             # run stage: only simulates (minutes); run_all.sh sets the kmeans/db stages' limits
#SBATCH --account=evgenios
#SBATCH --qos=normal
#SBATCH --array=0-191%64
#SBATCH --output=logs/Tiptoe_%A_%a.out
#SBATCH --error=logs/Tiptoe_%A_%a.err
#SBATCH --signal=B:TERM@120

# Tiptoe baseline (see baselines/README.md).
#   MODE=sim (default) baselines/tiptoe/simulate_tiptoe.py: exact quality + exact DB shapes + payload-only bytes,
#             no server-time benchmark (quality + bytes + LAN/WAN only). No SEAL needed.
#   MODE=real ./pir_app_tiptoe -t tiptoe, the real upstream protocol (needs `baselines/setup_baselines.sh tiptoe`).
#             Slow: SimplePIR hint generation over the whole corpus, then a 32 MiB token per query.
#
# Full grid search, nothing trimmed (unlike run_all_simulated.sh, which trims ma/bundle on big datasets locally):
#   DPC    docs per k-means cluster: fewer, bigger clusters = better recall, bigger download (Tiptoe used ~2500)
#   MA     clusters each boundary doc is copied into (Tiptoe: 2)
#   BUNDLE docs per URL bundle (Tiptoe: 160): bigger = more docs returned, bigger round-2 download
#   HINT   embedding hint size in MB (Tiptoe: 500): moves bytes between upload (DB width) and download (DB height)
#   k      10, 100
#   PREC   embedding precision, cast after loading (simulate_tiptoe.py --precision): fp32 fp8 fp1 by default
#          (fp16 fp4 fp2 also exist: TIPTOE_PRECISIONS="fp32 fp16 fp8 fp4 fp2 fp1" for all of them).
#          Same DBs for all of them, so it only multiplies the run stage. fp32 folders keep their old names, the
#          others get a _<prec> suffix. MODE=real only does fp32.
#   per dataset: 3 DPC x 2 MA x 2 BUNDLE x 2 HINT x 2 k x 3 PREC = 144 run tasks (the 48 fp32 ones skip at once
#   if they already have results); 4 datasets = 576.
# msmarco-v2 and its subsamples have their own script, run_tiptoe_msmarco_v2.slurm (longer preprocessing, one task
# per preprocessed DB instead of per config).
#
# Usage: submit with datasets/run_all.sh (STAGES=tiptoe), which chains the three stages below with afterok.
#   TIPTOE_DATASETS=nq TIPTOE_STAGE=kmeans|db|run bash run_tiptoe.slurm --count     # 3 / 12 / 48 per dataset
#
# Memory: preprocessing reads the embeddings memory-mapped (peak ~4 GB + <100 B per doc, measured on a 2M-doc set)
# and the simulator holds almost nothing; the simplepir_bench timing run is capped at 8 GB (--bench-max-gb, it
# times a slice and scales linearly; that run peaks at ~2x the cap). So sim mode stays under ~20 GB for every
# dataset here. MODE=real on msmarco: Tiptoe builds its SimplePIR DB unsquished first (~16 GB at 8.8M docs) plus
# the parsed corpus, so submit that with --mem=64G.
#
# Three stages (TIPTOE_STAGE), each a job array that waits for the one before (run_all.sh), so no task ever sits on
# an allocation waiting for another task's lock:
#   kmeans  one task per (dataset, DPC): the k-means (cached in tiptoe/kmeans_cache) + the first of its DBs
#   db      one task per (dataset, DPC, MA, BUNDLE): the bundling of that DB (~45 min for fever at DPC 250)
#   run     one task per config (+ HINT x k): only the simulation (minutes)
# A stage refuses to start if the stage before it hasn't finished (marker file / tiptoe_meta.json missing), instead
# of quietly doing that work itself; that path is what had dozens of tasks waiting on each other.

TIPTOE_DATASETS="${TIPTOE_DATASETS:-scifact nq fever msmarco}"
TIPTOE_STAGE="${TIPTOE_STAGE:-run}"   # kmeans | db | run, see above
case $TIPTOE_STAGE in kmeans|db|run) ;; *) echo "TIPTOE_STAGE must be kmeans, db or run" >&2; exit 1 ;; esac
DPC_LIST=(250 2500 40000)    # 2026-11-09, was 250 500 1000 2500 5000 10000 40000
MA_LIST=(1 2)
BUNDLE_LIST=(40 320)         # was 40 80 160 320
read -ra PREC_LIST <<< "${TIPTOE_PRECISIONS:-fp32 fp8 fp1}"   # 2026-11-14
[[ "${MODE:-sim}" == real ]] && PREC_LIST=(fp32)
DATA_ARR=(); DPC_ARR=(); MA_ARR=(); B_ARR=(); H_ARR=(); K_ARR=(); P_ARR=()
KM_D=(); KM_DPC=(); DB_D=(); DB_DPC=(); DB_MA=(); DB_B=()
for d in $TIPTOE_DATASETS; do
  for dpc in "${DPC_LIST[@]}"; do
    KM_D+=("$d"); KM_DPC+=("$dpc")
    for ma in "${MA_LIST[@]}"; do
      for bundle in "${BUNDLE_LIST[@]}"; do
        DB_D+=("$d"); DB_DPC+=("$dpc"); DB_MA+=("$ma"); DB_B+=("$bundle")
        for hint in 125 500; do
          for k in 10 100; do
            for prec in "${PREC_LIST[@]}"; do
              DATA_ARR+=("$d"); DPC_ARR+=("$dpc"); MA_ARR+=("$ma"); B_ARR+=("$bundle"); H_ARR+=("$hint"); K_ARR+=("$k")
              P_ARR+=("$prec")
            done
          done
        done
      done
    done
  done
done
if [[ "${1:-}" == "--count" ]]; then
    case $TIPTOE_STAGE in
        kmeans) echo "${#KM_D[@]} kmeans tasks for: $TIPTOE_DATASETS (use --array=0-$(( ${#KM_D[@]} - 1 )))" ;;
        db)     echo "${#DB_D[@]} db tasks for: $TIPTOE_DATASETS (use --array=0-$(( ${#DB_D[@]} - 1 )))" ;;
        run)    echo "${#DATA_ARR[@]} configs for: $TIPTOE_DATASETS (use --array=0-$(( ${#DATA_ARR[@]} - 1 ))%64)" ;;
    esac
    exit 0
fi

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
MODE="${MODE:-sim}"
if [[ "$MODE" == "real" ]]; then
    module load gnu12 2>/dev/null || module load gnu9/9.3.0   # pir_app_tiptoe was built with a C++20 compiler
else
    module load gnu9/9.3.0
fi

export NGT_PREFIX="$HOME/opt/ngt-amd"
export HNSW_PREFIX="$HOME/opt/hnsw"
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-1}"

source datasets/env.sh     # conda env main (finds conda itself; CONDA_BASE/CONDA_ENV to override)

task_id=$SLURM_ARRAY_TASK_ID
if [[ "$TIPTOE_STAGE" == kmeans ]]; then
    dataset=${KM_D[$task_id]}; dpc=${KM_DPC[$task_id]}; ma=${MA_LIST[0]}; bundle=${BUNDLE_LIST[0]}
elif [[ "$TIPTOE_STAGE" == db ]]; then
    dataset=${DB_D[$task_id]}; dpc=${DB_DPC[$task_id]}; ma=${DB_MA[$task_id]}; bundle=${DB_B[$task_id]}
else
    dataset=${DATA_ARR[$task_id]}
    dpc=${DPC_ARR[$task_id]}
    ma=${MA_ARR[$task_id]}
    bundle=${B_ARR[$task_id]}
    hint=${H_ARR[$task_id]}
    k_val=${K_ARR[$task_id]}
    prec=${P_ARR[$task_id]}
fi
method="tiptoe"

DATASET_DIR="${DATASET_DIR:-../../datasets}"
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/$USER/results}"

# -n and the real-mode query cap: hardcoded for the originals, num_docs.txt (datasets/prepare_dataset.py) otherwise
case "$dataset" in
    msmarco)    DB_SIZE=8841823; MAXQ=1000 ;;   # real mode: the per-query token is the slow part; 1000 is plenty
    scifact)    DB_SIZE=5183; MAXQ=0 ;;
    trec-covid) DB_SIZE=171332; MAXQ=0 ;;
    *)
        f="${DATASET_DIR}/${dataset}/num_docs.txt"
        [[ -f "$f" ]] || { echo "No $f, run datasets/prep_datasets.slurm for $dataset first" >&2; exit 1; }
        DB_SIZE=$(cat "$f"); MAXQ=1000 ;;
esac

FAIL_FILE="${SLURM_SUBMIT_DIR:-$PWD}/tiptoe_fails"
finished=0
trap '[[ $finished -eq 1 ]] || echo "${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} $(date "+%F %T")" >> "$FAIL_FILE"' EXIT
trap 'kill "${srun_pid:-}" 2>/dev/null; exit 143' TERM INT

# 1) Preprocess: the kmeans and db stages build the DBs, the run stage only reads them
TT_DIR="${DATASET_DIR}/tiptoe/${dataset}_dpc${dpc}_m${ma}_b${bundle}"
KM_DONE="${DATASET_DIR}/tiptoe/kmeans_cache/${dataset}_dpc${dpc}.done"   # written by the kmeans stage
need() {   # need <file> <stage that makes it>
    [[ -f "$1" ]] && return 0
    echo "ERROR: $1 is missing: the $2 stage for $dataset dpc=$dpc hasn't finished. Submit tiptoe through" >&2
    echo "       datasets/run_all.sh (STAGES=tiptoe), which runs kmeans -> db -> run in order." >&2
    exit 1
}
preprocess() {   # builds $TT_DIR (flock: a re-submitted task can overlap a running one)
    mkdir -p "$(dirname "$TT_DIR")" "${DATASET_DIR}/tiptoe/kmeans_cache"
    (
        flock 9
        if [[ ! -f "${TT_DIR}/tiptoe_meta.json" ]]; then
            python baselines/tiptoe/preprocess_tiptoe.py --dataset "$DATASET_DIR" --name "$dataset" \
                --docs-per-cluster "$dpc" --multi-assign "$ma" --bundle-size "$bundle" --out "$TT_DIR" \
                --kmeans-cache "${DATASET_DIR}/tiptoe/kmeans_cache"
        fi
    ) 9>"${TT_DIR}.lock"
}
case $TIPTOE_STAGE in
    kmeans)   # the k-means, by building this DPC's first DB (its centroids go to the cache)
        echo "kmeans: $dataset dpc=$dpc (via $TT_DIR)"
        preprocess; touch "$KM_DONE"; finished=1; exit 0 ;;
    db)       # one DB; its k-means is cached already
        need "$KM_DONE" kmeans
        echo "db: $TT_DIR"
        preprocess; finished=1; exit 0 ;;
esac
need "${TT_DIR}/tiptoe_meta.json" db

# multi-assign as applied (capped at the number of clusters): the simulator names its folder with it, so on a
# 1-cluster scifact config ma=2 lands in the m1 folder and is skipped here instead of racing the ma=1 task
read -r C MA_USED < <(python -c "import json;m=json.load(open('${TT_DIR}/tiptoe_meta.json'));print(m['num_clusters'],m['multi_assign'])")
PSFX=""; [[ "$prec" == fp32 ]] || PSFX="_$prec"     # same naming as simulate_tiptoe.py
OUT_DIR="${RESULTS_ROOT}/${method}_${dataset}_c${C}_m${MA_USED}_b${bundle}_h${hint}_k${k_val}${PSFX}"
# Several DPC values can give the same cluster count (scifact: 2500 and 40000 are both 1 cluster), so two tasks can
# share OUT_DIR; without this lock both simulated at once and interleaved metadata.json ("}}", the "bad json" in
# check_runs.sh). The second one now waits here (seconds) and then skips.
mkdir -p "$(dirname "$OUT_DIR")"
exec 8>"${OUT_DIR}.lock"
flock 8
if [[ -f "${OUT_DIR}/metadata.json" ]] && python -c "import json,sys; json.load(open(sys.argv[1]))" "${OUT_DIR}/metadata.json" 2>/dev/null; then
    # several DPC values can give the same cluster count (1 on scifact, the 25196 cap on msmarco-v2); first one wins
    echo "${OUT_DIR} already has results, skipping"; finished=1; exit 0
fi
mkdir -p "$OUT_DIR"

cat <<EOF | tee "${OUT_DIR}/slurm_name.txt"
========================================
Array Task ID:     $SLURM_ARRAY_TASK_ID
Method:            $method
Dataset:           $dataset
K Value:           $k_val
Docs per cluster:  $dpc  (clusters: $C)
Multi-assign:      $ma
Bundle size:       $bundle
Emb hint MB:       $hint
Precision:         $prec
Mode:              $MODE
Tiptoe dir:        $TT_DIR
Output Dir:        $OUT_DIR
========================================
EOF

# 2) Run
# Quality + communication (bytes -> LAN/WAN time) only. The server-time benchmark (simplepir_bench) is never run.
if [[ "$MODE" == "sim" ]]; then
    srun python baselines/tiptoe/simulate_tiptoe.py \
        --dataset "$DATASET_DIR" --name "$dataset" --tiptoe-dir "$TT_DIR" --k "$k_val" \
        --emb-hint-mb "$hint" --results-root "$RESULTS_ROOT" --precision "$prec" \
        --parallelism "${SLURM_CPUS_PER_TASK:-1}" &
    srun_pid=$!
    wait "$srun_pid"
    mv "logs/Tiptoe_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/execution.log" || true
    mv "logs/Tiptoe_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/error.log" || true
    finished=1
    echo "All done for this node."
    exit 0
fi

srun ./pir_app_tiptoe \
    -n "$DB_SIZE" \
    -t "$method" \
    -name "$dataset" \
    -k "$k_val" \
    -tiptoe-dir "$TT_DIR" \
    -tiptoe-max-queries "$MAXQ" \
    -tiptoe-emb-hint-mb "$hint" \
    -outDir "$OUT_DIR" \
    -outFile "results" \
    -dataset "$DATASET_DIR" \
    -debug 0 \
    -dim 192 &
srun_pid=$!
wait "$srun_pid"

mv "logs/Tiptoe_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/execution.log" || true
mv "logs/Tiptoe_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/error.log" || true
finished=1
echo "All done for this node."
