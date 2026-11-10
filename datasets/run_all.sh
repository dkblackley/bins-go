#!/bin/bash
# Queue the whole pipeline for one or more datasets with a single command. Every job waits for the jobs it needs
# (sbatch --dependency=afterok), so you can submit everything now and walk away.
#
#   bash datasets/run_all.sh nq hotpotqa fever
#   bash datasets/run_all.sh msmarco-v2-all          # msmarco-v2, then the 25m/50m/100m subsamples once it's embedded
#   STAGES="bins pacmann" bash datasets/run_all.sh nq      # only some stages (data already prepared)
#   DRY_RUN=1 bash datasets/run_all.sh nq            # print the sbatch commands, submit nothing
#
# Stages, in order (default: all):
#   prep     download + convert                 datasets/prep_datasets.slurm
#   embed    e5 + PCA embeddings                 datasets/embed.slurm (msmarco-v2: PCA -> shards -> merge)
#   tree-db  Lucene index + tree artifacts       datasets/tree_artifacts.slurm          (after embed)
#   bins-index Bluge index the bins need         datasets/bins_index.slurm              (after prep)
#   bins     bins grid, high + low tiers         run_bins_high.slurm, run_bins_low.slurm (after embed + bins-index)
#   pacmann  pacmann grid                        run_pacmann.slurm                       (after embed)
#   tree     tree grid                           run_tree_pir_old.slurm                  (after tree-db)
#   tiptoe   Tiptoe baseline (simulated: quality + bytes, no server time)                     (after embed)
#            run_tiptoe.slurm, or run_tiptoe_msmarco_v2.slurm for msmarco-v2*. Not in the default STAGES.
#   coeus    Coeus baseline (quality + bytes, no server time) run_coeus.slurm               (after prep)
#            Not in the default STAGES. Both skip configs that already have a metadata.json.
#            SWEEP=legacy|oat|grid|all picks the tree grid (tree_sweep_grid.sh, default all). msmarco/scifact
#            already have their data, so for them use STAGES="tree-db tree". Tree runs are split by estimated RAM:
#            <=100 GB on the normal partition, 250/500/1000/1500 GB on bigmem, one job array per bucket.
#
# SIMULATE=1 runs bins, pacmann and tree with pir_app -simulate (no hints, no XOR: same answers, MRR, bytes, LAN/WAN
# and client storage; every timing -1) into $RESULTS_DIR, default /scratch/$USER/results_simulated, with their own
# usage log and smaller first estimates (slurm_common.sh). The data stages and tiptoe/coeus are skipped then.
#   SIMULATE=1 STAGES=tree bash datasets/run_all.sh nq dbpedia-entity fever msmarco
# SIMULATE=1 NORAM=1 (what run_sim_noram.sh sets) also passes -noram and sizes the jobs for simulated low-RAM runs
# (a few GB, minutes; floor 12 GB / 30 min): see run_sim_noram.sh.
# MAX_GB=n skips every bins/tree group estimated above n GB (e.g. MAX_GB=100: only the normal partition).
#
# GPU=<profile> sends the embed jobs to another GPU queue instead of embed.slurm's DGX A100.40gb (fp16, short time
# limits, 32 GB RAM, so they backfill sooner; a job that hits its limit requeues itself and resumes). Run it next to a series that's already queued: whichever embed finishes
# first wins, the other one exits 3 and its dependants stay pending (DependencyNeverSatisfied) for you to scancel.
#   a100       A100.80gb on gpuq OR contrib-gpuq (96 GPUs; preemptible on contrib, jobs requeue + resume)   <- default pick
#   any        any GPU at all (--gres=gpu:1 on gpuq, contrib-gpuq, contrib-h100, incl. MIG slices), 4 h limit
#   a100-gpuq  A100.80gb on gpuq only (40 GPUs, never preempted)
#   mig40      MIG 3g.40gb slice on gpuq (24; ~0.4 of an A100)      mig20  MIG 2g.20gb (8; ~0.25 of an A100)
#   h100       H100.80gb on contrib-h100 (4, preemptible)           b200   B200.180gb on contrib-b200 (8, reserved)
#   dgx        A100.40gb on gpuq (what embed.slurm asks for), but fp16 + the shorter limits
# Job names get a _<profile> suffix (squeue -u $USER -o "%.10i %.9P %.28j %.2t %.10M %R" shows them in full).
# A stage you leave out is assumed to be done already. If a job fails, the jobs that depend on it stay pending
# with reason DependencyNeverSatisfied: fix it, scancel them and re-run with STAGES set to the remaining stages.
#
# Run from the repo root after build.sh (pir_app must exist). Array sizes are worked out with each script's --count.
set -euo pipefail
cd "$(dirname "$0")/.."

# One dataset folder for every stage (default ../datasets, i.e. run from the scratch copy of the repo). The run
# scripts on their own default to ../../datasets, so pin both here (override with DATASETS_ROOT=...).
export DATASETS_ROOT="${DATASETS_ROOT:-../datasets}"
export DATASET_DIR="$DATASETS_ROOT"

STAGES="${STAGES:-prep embed tree-db bins-index bins pacmann tree}"
has() { [[ " $STAGES " == *" $1 "* ]]; }
mkdir -p logs

submit() {   # submit <dependency job ids, space separated, may be empty> <sbatch args...> -> prints job id
    # a job id prefixed "any:" is an afterany dependency (run once it ended, however), the rest afterok
    local deps="$1"; shift
    local dep_arg=() ok="" any="" j
    for j in $deps; do
        if [[ $j == any:* ]]; then any+="${any:+:}${j#any:}"; else ok+="${ok:+:}$j"; fi
    done
    local spec="${ok:+afterok:$ok}"
    [[ -n "$any" ]] && spec+="${spec:+,}afterany:$any"
    [[ -n "$spec" ]] && dep_arg=(--dependency="$spec" --kill-on-invalid-dep=no)
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "  sbatch ${dep_arg[*]} $*" >&2
        echo "DRY$RANDOM"
    else
        sbatch --parsable "${dep_arg[@]}" "$@"
    fi
}

count_of() {   # count_of <env assignment> <script> -> number of array tasks
    env "$1" SLURM_SUBMIT_DIR="$PWD" bash "$2" --count 2>/dev/null | grep -oE '^[0-9]+' || echo 0
}

# ---- GPU profile for the embed jobs (see header). Prints sbatch args for one embed MODE.
SFX="${GPU:+_$GPU}"     # job name suffix, so two series are told apart in squeue
[[ "${SIMULATE:-0}" == 1 ]] && SFX+="_sim"   # simulated jobs: queued_dirs keeps them apart from real ones
gpu_args() {   # gpu_args <all|pca|shard|finish>
    [[ -z "${GPU:-}" ]] && return 0
    local part qos=gpu gres requeue="" cpus=16 t_all t_shard
    case "$GPU" in
        any)       part=gpuq,contrib-gpuq,contrib-h100; gres="";  cpus=4; requeue=1; t_all=4; t_shard=4 ;;
        a100)      part=gpuq,contrib-gpuq; gres=A100.80gb;  requeue=1; t_all=2;  t_shard=2 ;;
        a100-gpuq) part=gpuq;              gres=A100.80gb;             t_all=2;  t_shard=2 ;;
        dgx)       part=gpuq;              gres=A100.40gb;             t_all=2;  t_shard=2 ;;
        mig40)     part=gpuq;              gres=3g.40gb;    cpus=4;    t_all=6;  t_shard=6 ;;
        mig20)     part=gpuq;              gres=2g.20gb;    cpus=4;    t_all=10; t_shard=10 ;;
        h100)      part=contrib-h100;      gres=H100.80gb;  requeue=1; t_all=2;  t_shard=2 ;;
        b200)      part=contrib-b200;      gres=B200.180gb; requeue=1; t_all=2;  t_shard=2 ;;
        *) echo "Unknown GPU=$GPU (any a100 a100-gpuq dgx mig40 mig20 h100 b200)" >&2; exit 1 ;;
    esac
    part="${GPU_PARTITION:-$part}"; gres="${GPU_GRES:-$gres}"; qos="${GPU_QOS:-$qos}"
    local hours mem=32G
    case "$1" in
        all) hours=$t_all ;; shard) hours=$t_shard ;; pca) hours=1 ;; finish) hours=4; mem=64G ;;
    esac
    hours="${GPU_HOURS:-$hours}"
    echo "--partition=$part --qos=$qos --gres=gpu:${gres:+$gres:}1 --cpus-per-task=$cpus --mem=$mem --time=$(printf '0-%02d:00:00' "$hours") --open-mode=append ${requeue:+--requeue}"
}
# fp16 + a per-series resume key for the GPU= series (embed.slurm / collection_preproc.py read these)
EMB_ENV=""
[[ -n "${GPU:-}" ]] && EMB_ENV=",EMBED_FP16=${EMBED_FP16:-1},EMBED_TAG=${GPU}"

is_big() { [[ "$1" == msmarco-v2* || "$1" == miracl-en ]]; }

source ./slurm_common.sh
ONLY_IF_MISSING=""   # stages queued only when their outputs are missing (set for SIMULATE=1 below)
if [[ "${SIMULATE:-0}" == 1 ]]; then
    echo "SIMULATE=1: simulated PIR (pir_app -simulate, every timing -1), results in $RESULTS_DIR"
    for s in prep embed tiptoe coeus; do
        has "$s" && { echo "  (stage $s is not simulated; it is skipped here, run it without SIMULATE)"; STAGES=" ${STAGES/$s/} "; }
    done
    # The tree needs its artifacts (tree-db) and the bins their Bluge index (bins-index). Both are the same for real
    # and simulated runs, so they are queued here too, but only for a dataset where something is missing; the
    # simulated runs then wait for them (afterok). prep_tree.sh skips the B/r/s shapes that already exist.
    has tree && ! has tree-db && STAGES+=" tree-db"
    has bins && ! has bins-index && STAGES+=" bins-index"
    ONLY_IF_MISSING=" tree-db bins-index "
fi
only_if_missing() { [[ "$ONLY_IF_MISSING" == *" $1 "* ]]; }
# upstream_missing <file>: true if the file comes from prep/embed (which SIMULATE=1 never queues), not tree-db/bins-index
upstream_missing() { [[ "$1" == *_embed_f32.npy || "$1" == */num_docs.txt || "$1" == */corpus.jsonl || "$1" == */queries*.jsonl || "$1" == */qrels/* ]]; }
LIST_DIR="logs/lists/$(date +%Y%m%d_%H%M%S)"   # CONFIG_LIST files, kept so you can see what each array runs
mkdir -p "$LIST_DIR"
# Configs a job in the queue already covers are skipped (slurm_common.sh queued_dirs), so re-running this after a
# fix only submits what's missing. Worked out once here; run_me.sh passes its own QUEUED_FILE.
export QUEUED_FILE="${QUEUED_FILE:-$LIST_DIR/queued.tsv}"
[[ -f "$QUEUED_FILE" ]] || queued_dirs > "$QUEUED_FILE"
if grep -q '^?' "$QUEUED_FILE"; then
    echo "NOTE: can't tell which configs these queued arrays run (no CONFIG_LIST in sacct's SubmitLine), so nothing is"
    echo "      submitted for their method + dataset until they leave the queue:"
    awk -F'\t' '$1 == "?" { print "        " $2 " " $3 " (job " $4 ")" }' "$QUEUED_FILE" | sort -u
fi

# true if any of these upstream jobs is being queued now (their outputs don't exist yet, so don't check inputs)
inputs_pending() { [[ -n "$(echo $1)" ]]; }

# pacmann_todo <dataset> <grid size> -> slurm --array list (e.g. "0-3,7,9-71") of the configs still to run; same
# order as run_pacmann.slurm: k (10 100) x steps (5..30) x neighbours (8..48). FORCE=1 lists every config.
pacmann_todo() {
    local d=$1 n=$2 i k st nb od res="${RESULTS_DIR:-/scratch/${USER:-$(id -un)}/results}" idx=()
    load_queued
    for (( i = 0; i < n; i++ )); do
        k=$(( i < 36 ? 10 : 100 )); st=$(( (i % 36 / 6 + 1) * 5 )); nb=$(( (i % 6 + 1) * 8 ))
        od="$res/pacmann_${d}_k${k}_steps${st}_neighb${nb}"
        [[ -z "${FORCE:-}" && -s "$od/metadata.json" ]] && continue
        is_queued pacmann "$d" "$od" && continue
        idx+=("$i")
    done
    printf '%s\n' "${idx[@]}" | awk 'NF { if ($1 == p + 1 && c) { p = $1; next }
        if (c) out = out (s == p ? s : s "-" p) ","; s = p = $1; c = 1 }
        END { if (c) out = out (s == p ? s : s "-" p); print out }'
}
pacmann_count() { awk -v a="$1" 'BEGIN { n = split(a, P, ","); for (i = 1; i <= n; i++) { split(P[i], r, "-"); t += (r[2] == "" ? 1 : r[2] - r[1] + 1) } print t }'; }

# submit_groups <method> <dataset> <plan file> <dependencies> <script for normal> <script for bigmem>
GROUP_JIDS=""   # the job ids the last submit_groups call submitted
submit_groups() {   # submit_groups method dataset plan deps normal_script big_script [list prefix]
    local method=$1 d=$2 plan=$3 deps=$4 s_normal=$5 s_big=$6 gb min n f chunk=${MAX_ARRAY:-1000} off
    local prefix="${7:-${method}_${d}}"
    local total; total=$(grep -c . "$plan" || true)
    GROUP_JIDS=""
    while read -r gb min n f; do
        case $gb in
            missing) echo "  $method  SKIPPED $min of $total configs: inputs not there yet (first missing: $n)"; continue ;;
            over)    echo "  $method  SKIPPED $min configs: estimated above ${MEM_BUCKETS##* } GB"; continue ;;
            done)    echo "  $method  SKIPPED $min of $total configs: already finished (metadata.json; FORCE=1 reruns them)"; continue ;;
            queued)  echo "  $method  SKIPPED $min of $total configs: a job in the queue already runs them"; continue ;;
        esac
        if [[ -n "${MAX_GB:-}" ]] && (( ${gb%.*} > MAX_GB )); then   # MAX_GB=n: leave the biggest configs for later
            echo "  $method  SKIPPED $n configs: estimated ${gb}G, above MAX_GB=$MAX_GB"; continue
        fi
        local part script
        part=$(partition_for "$gb"); [[ $part == normal ]] && script=$s_normal || script=$s_big
        for (( off = 0; off < n; off += chunk )); do
            local m=$(( n - off < chunk ? n - off : chunk )) jid
            jid=$(submit "$deps" --export=ALL,CONFIG_LIST="$f",ARRAY_OFFSET=$off --array=0-$((m - 1)) \
                 --partition="$part" --mem="${gb}G" --time="$(slurm_time "$min")" \
                 --job-name="${method}_${d}_${gb}G$SFX" "$script")
            # which list + offset this array runs, so queued_dirs (slurm_common.sh) can map its tasks to configs
            [[ -n "${DRY_RUN:-}" ]] || printf '%s\t%s\t%s\n' "$jid" "$f" "$off" >> "$SUBMITTED_FILE"
            echo "  $method  $jid (${gb}G ${min}min $part: $m configs, $(basename "$f"))"
            GROUP_JIDS+=" $jid"
        done
    done < <(group_plan "$plan" "$LIST_DIR" "$prefix")
}

# submit_dataset <name> [extra dependency] ; sets EMBED_JOB to the job the dataset's embeddings come from
submit_dataset() {
    local d="$1" extra="${2:-}" prep="" embed="" treedb="" binsidx=""
    echo "== $d"

    # a msmarco-v2 subsample's "prep" copies the full set's embedding rows, so it's part of its embed stage too
    if has prep || { has embed && [[ "$d" =~ ^msmarco-v2-[0-9]+m$ ]]; }; then
        prep=$(submit "$extra" --export=ALL,DATASET="$d" --job-name="prep_$d$SFX" datasets/prep_datasets.slurm)
        echo "  prep     $prep"
    fi

    embed="$prep $extra"
    local pfx="$d"; [[ "$d" == trec-covid ]] && pfx=trec
    if has embed && [[ -f "$DATASETS_ROOT/$d/${pfx}_corpus_embed_f32.npy.done" ]]; then
        echo "  embed    already done ($d/${pfx}_corpus_embed_f32.npy.done), not queued"
    elif has embed && [[ ! "$d" =~ ^msmarco-v2-[0-9]+m$ ]]; then   # subsamples copy the full set's embeddings
        if [[ "$d" == "msmarco-v2" || "$d" == "miracl-en" ]]; then
            local n=${NUM_SHARDS:-$([[ "$d" == miracl-en ]] && echo 8 || echo 24)} pca shards
            pca=$(submit "$prep $extra" --export=ALL,DATASET="$d",MODE=pca,NUM_SHARDS=$n$EMB_ENV --time=0-02:00:00 \
                  $(gpu_args pca) --job-name="pca_$d$SFX" datasets/embed.slurm)
            shards=$(submit "$pca" --export=ALL,DATASET="$d",MODE=shard,NUM_SHARDS=$n$EMB_ENV --array=0-$((n - 1)) \
                  $(gpu_args shard) --job-name="emb_$d$SFX" datasets/embed.slurm)
            embed=$(submit "$shards" --export=ALL,DATASET="$d",MODE=finish,NUM_SHARDS=$n$EMB_ENV --time=0-06:00:00 \
                  $(gpu_args finish) --job-name="merge_$d$SFX" datasets/embed.slurm)
            echo "  embed    pca=$pca shards=$shards merge=$embed"
        else
            embed=$(submit "$prep $extra" --export=ALL,DATASET="$d"$EMB_ENV $(gpu_args all) --job-name="emb_$d$SFX" \
                  datasets/embed.slurm)
            echo "  embed    $embed"
        fi
    fi
    EMBED_JOB="$embed"

    local tdb_miss="" bidx_miss=""
    if has tree-db && only_if_missing tree-db; then
        tdb_miss=$(DATASET="$d" SWEEP="${SWEEP:-all}" bash tree_sweep_grid.sh --missing 2>/dev/null || true)
        if [[ -z "$tdb_miss" ]]; then
            echo "  tree-db  artifacts all there, not queued"
        elif upstream_missing "$tdb_miss"; then
            echo "  tree-db  NOT queued: $tdb_miss is missing, which comes from prep/embed (run those without SIMULATE)"
            tdb_miss=""
        fi
    fi
    if has bins-index && only_if_missing bins-index; then
        bidx_miss=$(bins_missing "$d")
        if [[ -z "$bidx_miss" ]]; then
            echo "  bins-index already built, not queued"
        elif upstream_missing "$bidx_miss"; then
            echo "  bins-index NOT queued: $bidx_miss is missing, which comes from prep/embed (run those without SIMULATE)"
            bidx_miss=""
        fi
    fi

    if has tree-db && { ! only_if_missing tree-db || [[ -n "$tdb_miss" ]]; }; then
        [[ -n "$tdb_miss" ]] && echo "  tree-db  queued because $tdb_miss is missing"
        local big=()
        is_big "$d" && big=(--partition=bigmem --mem=1000G)
        treedb=$(submit "$embed" --export=ALL,DATASET="$d",SWEEP="${SWEEP:-all}",WORKERS="${WORKERS:-$(is_big "$d" && echo 12 || echo 16)}" \
                 "${big[@]}" --job-name="treedb_$d$SFX" datasets/tree_artifacts.slurm)
        echo "  tree-db  $treedb"
    fi

    if has bins-index && { ! only_if_missing bins-index || [[ -n "$bidx_miss" ]]; }; then
        [[ -n "$bidx_miss" ]] && echo "  bins-index queued because $bidx_miss is missing"
        local big=()
        is_big "$d" && big=(--partition=bigmem --mem=500G)
        binsidx=$(submit "$prep $extra" --export=ALL,DATASET="$d" "${big[@]}" --job-name="binsidx_$d$SFX" \
                  datasets/bins_index.slurm)
        echo "  bins-index $binsidx"
    fi

    if has bins; then
        # every config of both tiers, estimated (measured usage if any, else the model in slurm_common.sh), split
        # into one array per (memory, time) bucket: <=100 GB -> run_bins_low.slurm on normal, else run_bins_high on bigmem
        local plan="$LIST_DIR/bins_${d}.plan"
        { BINS_DATASETS="$d" SLURM_SUBMIT_DIR="$PWD" bash run_bins_low.slurm --list-raw
          BINS_DATASETS="$d" SLURM_SUBMIT_DIR="$PWD" bash run_bins_high.slurm --list-raw; } \
            | CHECK_INPUTS=$(inputs_pending "$prep $embed $binsidx" && echo 0 || echo 1) bins_plan > "$plan"
        submit_groups bins "$d" "$plan" "$embed $binsidx" run_bins_low.slurm run_bins_high.slurm
    fi

    if has pacmann; then
        # only the configs without a metadata.json that no queued job covers (task i = config i of the grid)
        local n big=() todo
        n=$(count_of PACMANN_DATASETS="$d" run_pacmann.slurm)
        todo=$(pacmann_todo "$d" "$n")
        if [[ "${SIMULATE:-0}" == 1 ]]; then
            # simulated: no hints, sized from the dataset (slurm_common.sh pacmann_sim_model), not the 120 GB / 5 h default
            local pg pm; read -r pg pm <<< "$(pacmann_sim_model "$d")"
            pg=$(mem_bucket "$pg"); pm=$(time_bucket "$pm")
            big=(--partition="$(partition_for "$pg")" --mem="${pg}G" --time="$(slurm_time "$pm")")
        elif is_big "$d"; then
            big=(--partition=bigmem --mem=1500G)
        fi
        if [[ -z "$todo" ]]; then
            echo "  pacmann  SKIPPED: all $n configs finished or already queued"
        else
            echo "  pacmann  $(submit "$embed" --export=ALL,PACMANN_DATASETS="$d" --array="$todo" "${big[@]}" \
                 --job-name="pacmann_$d$SFX" run_pacmann.slurm) ($(pacmann_count "$todo") of $n tasks: $todo)"
        fi
    fi

    if has tree; then
        # every config of the sweep, estimated and split by (memory, time) like bins; one script for both partitions
        local plan="$LIST_DIR/tree_${d}.plan"
        ( DATASET="$d" SWEEP="${SWEEP:-all}" CHECK_INPUTS=$(inputs_pending "$prep $embed $treedb" && echo 0 || echo 1) \
            bash tree_sweep_grid.sh --plan ) > "$plan"
        # Shared Cuckoo cache (tree_plan): the configs that build a shape's tables, and those whose shape is already
        # cached, go first; the rest of each uncached shape are sized as cache hits and wait for the build jobs
        # (afterany). Different list prefixes: group_plan rewrites its list files, and queued jobs still read theirs.
        awk '$(NF-1) !~ /^after:/' "$plan" > "$plan.first"
        awk '$(NF-1) ~ /^after:/' "$plan" > "$plan.after"
        submit_groups tree "$d" "$plan.first" "$treedb $embed" run_tree_pir_old.slurm run_tree_pir_old.slurm "tree_${d}"
        local tree_build_jids="$GROUP_JIDS"
        if [[ -s "$plan.after" ]]; then
            local any="" j
            for j in $tree_build_jids; do any+=" any:$j"; done
            echo "  tree     $(grep -c . "$plan.after") configs of shapes being built now: sized as cache hits, after${any:- (no build jobs submitted)}"
            submit_groups tree "$d" "$plan.after" "$treedb $embed$any" run_tree_pir_old.slurm run_tree_pir_old.slurm "tree_${d}_after"
        fi
    fi

    if has tiptoe; then
        # chained stages (each afterok on the one before), so no task holds an allocation waiting on another's lock:
        #   small sets: kmeans (1 per DPC) -> db (1 per DB) -> run (1 per config, simulation only)
        #   msmarco-v2*: kmeans (1 per cluster count) -> run (1 per DB: bundling + its 4 simulations)
        # Limits from the logs: one fever DB bundles in ~45 min, a simulation takes minutes.
        local tk td n
        if [[ "$d" == msmarco-v2* ]]; then
            local t=0-06:00:00; [[ "$d" == msmarco-v2 ]] && t=0-12:00:00   # 138M docs: assignment + bundling ~2-4 h+
            v2count() { env TIPTOE_V2_DATASETS="$d" TIPTOE_STAGE=$1 SLURM_SUBMIT_DIR="$PWD" bash run_tiptoe_msmarco_v2.slurm --count 2>/dev/null | grep -oE '^[0-9]+' || echo 0; }
            tk=$(submit "$embed" --export=ALL,TIPTOE_V2_DATASETS="$d",TIPTOE_STAGE=kmeans --array=0-$(( $(v2count kmeans) - 1 )) \
                 --time=$t --job-name="tiptoekm_$d$SFX" run_tiptoe_msmarco_v2.slurm)
            n=$(v2count run)
            echo "  tiptoe   kmeans $tk, then $(submit "$tk" --export=ALL,TIPTOE_V2_DATASETS="$d",TIPTOE_STAGE=run \
                 --array=0-$((n - 1))%8 --time=$t --job-name="tiptoe_$d$SFX" run_tiptoe_msmarco_v2.slurm) ($n DBs x 4 configs)"
        else
            ttcount() { env TIPTOE_DATASETS="$d" TIPTOE_STAGE=$1 SLURM_SUBMIT_DIR="$PWD" bash run_tiptoe.slurm --count 2>/dev/null | grep -oE '^[0-9]+' || echo 0; }
            tk=$(submit "$embed" --export=ALL,TIPTOE_DATASETS="$d",TIPTOE_STAGE=kmeans --array=0-$(( $(ttcount kmeans) - 1 )) \
                 --time=0-04:00:00 --job-name="tiptoekm_$d$SFX" run_tiptoe.slurm)
            td=$(submit "$tk" --export=ALL,TIPTOE_DATASETS="$d",TIPTOE_STAGE=db --array=0-$(( $(ttcount db) - 1 )) \
                 --time=0-03:00:00 --job-name="tiptoedb_$d$SFX" run_tiptoe.slurm)
            n=$(ttcount run)
            echo "  tiptoe   kmeans $tk, db $td, then $(submit "$td" --export=ALL,TIPTOE_DATASETS="$d",TIPTOE_STAGE=run \
                 --array=0-$((n - 1))%64 --time=0-01:00:00 --job-name="tiptoe_$d$SFX" run_tiptoe.slurm) ($n configs)"
        fi
    fi

    if has coeus; then
        # A prep job tokenises the corpus once (COEUS_STAGE=prep, cached); every F/tf/k config then only builds its
        # postings (~12 B per (doc, term): ~5 GB on msmarco, ~50 GB on msmarco-v2) and fails if the cache is missing.
        local n prep_res=(--time=0-04:00:00) run_res=(--time=0-02:00:00) cprep
        n=$(count_of COEUS_DATASETS="$d" run_coeus.slurm)
        case $d in
            msmarco-v2)     prep_res=(--mem=120G --time=2-00:00:00); run_res=(--mem=120G --time=0-12:00:00) ;;
            msmarco-v2-*)   prep_res=(--mem=64G --time=1-00:00:00);  run_res=(--mem=64G --time=0-06:00:00) ;;
        esac
        cprep=$(submit "$embed" --export=ALL,COEUS_DATASETS="$d",COEUS_STAGE=prep --array=0 "${prep_res[@]}" \
                --job-name="coeusprep_$d$SFX" run_coeus.slurm)
        echo "  coeus    prep $cprep, then $(submit "$cprep" --export=ALL,COEUS_DATASETS="$d" --array=0-$((n - 1)) \
             "${run_res[@]}" --job-name="coeus_$d$SFX" run_coeus.slurm) ($n configs)"
    fi
}

[[ $# -gt 0 ]] || { sed -n 2,32p "$0"; exit 1; }
needs_app() { has bins-index || has bins || has pacmann || has tree; }
[[ -x ./pir_app || -n "${DRY_RUN:-}" ]] || ! needs_app || echo "WARNING: ./pir_app not found, run build.sh before the run jobs start" >&2

for d in "$@"; do
    if [[ "$d" == "msmarco-v2-all" ]]; then
        submit_dataset msmarco-v2
        full="$EMBED_JOB"
        for sub in msmarco-v2-25m msmarco-v2-50m msmarco-v2-100m; do
            submit_dataset "$sub" "$full"
        done
    else
        submit_dataset "$d"
    fi
done
echo "Queued. Watch with: squeue -u \$USER"
