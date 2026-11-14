#!/bin/bash
# make_snapshot.sh: build the anonymized copy of this repo for double-blind review (e.g. anonymous.4open.science).
#
#   bash tools/anon/make_snapshot.sh                 # -> ../pillar-anon (must not exist yet)
#   bash tools/anon/make_snapshot.sh /tmp/pillar-anon
#   ALLOW_DIRTY=1 bash tools/anon/make_snapshot.sh   # snapshot HEAD even with uncommitted changes (they are NOT included)
#
# What it does, on a copy (this checkout is never changed):
#   1. exports the committed tree at HEAD with `git archive`: no .git, no history, no untracked or ignored files;
#   2. deletes the files in EXCLUDE below (cluster-operations helpers, legacy scripts, debug tools) and tools/ itself,
#      which holds the identifying strings this script replaces;
#   3. swaps in tools/anon/README.md as the README;
#   4. rewrites identifying strings (Go module path, cluster and account names, home/scratch paths, author name);
#   5. fails if any identifying string is left (LEAK_RE), and checks the Go code still builds.
#
# Upload the output folder itself (or a zip of it), never a git remote: the git history has the author's name in
# every commit. Edit EXCLUDE / the rules below and re-run as often as needed.
set -euo pipefail

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
OUT="${1:-$REPO/../pillar-anon}"
MODULE_NEW="${MODULE_NEW:-pillar}"

[[ -e "$OUT" ]] && { echo "error: $OUT already exists; remove it or pick another folder" >&2; exit 1; }
if [[ -z "${ALLOW_DIRTY:-}" ]] && [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]]; then
    echo "error: uncommitted changes in $REPO (only HEAD is exported). Commit them, or ALLOW_DIRTY=1." >&2; exit 1
fi

# ---------------------------------------------------------------- 1. export HEAD
mkdir -p "$OUT"
git -C "$REPO" archive --format=tar HEAD | tar -x -C "$OUT"
cd "$OUT"

# ---------------------------------------------------------------- 2. files left out of the anonymous copy
# They stay in the real repo. Nothing the experiments need is listed here.
EXCLUDE=(
    # author's cluster-operations helpers (queue status, holds, resubmission, cache cleaning)
    hold_jobs.sh shrink_time.sh check_estimates.sh clean_logs.sh
    datasets/gpu_free.sh datasets/check_treedb.slurm datasets/link_tree_originals.sh
    datasets/colab_msmarco_v2.ipynb
    # legacy pipeline, superseded by run_tree_pir_old.slurm / datasets/*
    run_tree.slurm big_bins.slurm pre_proc_vec.slurm run_test.sh
    run_convert.sh filter_queries.py do_tree_MRR_Recall.py plotting/do_post.sh json_to_tsv.py re_rank.py
    # collaborator's original tree scripts with their own machine paths, and one-off debugging tools
    tree/README.md tree/run_python.sh tree/run_go.sh tree/SLURM_go.slurm
    tree/preprocess_msmarco.py tree/preprocess_queries.py
    tree/tools/align_embedding.py tree/tools/check_gold_subblock.py tree/tools/check_zero.py
    tree/tools/diag_stage2_recall.py tree/tools/find_block_score.py tree/tools/root_cause_analysis.py
    tree/tools/shuffle_block.py tree/tools/stage1_block_score.py tree/tools/stage2_row_inspect.py
)
rm -rf -- "${EXCLUDE[@]}"

# ---------------------------------------------------------------- 3. README
cp tools/anon/README.md README.md
rm -rf tools

# ---------------------------------------------------------------- 4. rewrite identifying strings
# Order matters: specific rules first, then the generic ones.
mapfile -t TEXT_FILES < <(grep -rIl --exclude-dir=third_party '' . || true)
mapfile -t GOMODS < <(find . -name go.mod -not -path './third_party/*')
export MODULE_NEW
perl -0pi -e '
    s{https://github\.com/dkblackley/bins-go(\.git)?}{<anonymous repository URL>}g;
    s{github\.com/dkblackley/bins-go}{$ENV{MODULE_NEW}}g;
    s{^#SBATCH --account=evgenios[^\n]*\n}{}mg;
    s{/projects/evgenios/dblackle/miniforge}{\$HOME/miniforge}g;
    s{/projects/evgenios/\$USER/}{\$HOME/}g;
    s{/home/dblackle/}{\$HOME/}g;
    s{/scratch/dblackle\b}{/scratch/\$USER}g;
    s{/home/mingxunz/Private-Search/}{/path/to/}g;
    s{/media/tson1997/}{/path/to/}g;
    s{https://drive\.google\.com/\S+}{<link withheld for anonymous review>}g;
    s{Copyright \(c\) (\d{4}) Daniel Blackley}{Copyright (c) $1 Anonymous Authors}g;
    s{^([ \t]*)module load hosts/hopper[ \t]*(?:2>/dev/null \|\| true)?[ \t]*$}{$1# module load <your site modules>}mg;
    s{module load hosts/hopper }{module load }g;
    s{results_hopper}{results_cluster}g;
    s{\bhopper:}{cluster:}g;
    s{\(Hopper\)}{(cluster)}g;
    s{\byour Hopper\b}{your cluster}g;
    s{\b[Hh]opper\x27s\b}{the cluster\x27s}g;
    s{hosts/hopper}{<site modules>}g;
    s{\b[Hh]opper\b}{the cluster}g;
' "${TEXT_FILES[@]}"
# undo the doubled article the generic rule can leave ("the the cluster")
perl -pi -e 's{\bthe the cluster\b}{the cluster}g; s{  the cluster}{ the cluster}g' "${TEXT_FILES[@]}"

# ---------------------------------------------------------------- 5. checks
LEAK_RE='dkblackley|dblackle|blackley|yelnat|daniel|evgenios|hopper|mingxunz|tson1997|drive\.google|fedora-gpd|claude|@gmail|claude\.ai'
if leaks="$(grep -rInEi --exclude-dir=third_party "$LEAK_RE" . | cut -c1-200)" && [[ -n "$leaks" ]]; then
    echo "LEFT OVER identifying strings (fix the rules above, or the files):" >&2
    echo "$leaks" >&2
    exit 2
fi
# binaries / data that should not be uploaded
big="$(find . -type f -size +1M -not -name '*.faiss' -not -path './third_party/*' -print)"
[[ -n "$big" ]] && { echo "warning: files over 1 MB:"; echo "$big"; }

echo "module path -> $MODULE_NEW; building the Go packages that need no native libs ..."
pkgs="$(go list ./... | grep -v -e '/Pacmann' -e "^$MODULE_NEW\$")"
# shellcheck disable=SC2086
go build $pkgs
if [[ -f "$HOME/opt/pacmann_env.sh" ]]; then
    source "$HOME/opt/pacmann_env.sh"
    go build -o /dev/null ./main.go && echo "pir_app builds" || { echo "error: pir_app does not build" >&2; exit 3; }
else
    echo "(NGT/hnsw not set up here, no ~/opt/pacmann_env.sh: skipped building pir_app itself)"
fi

echo
echo "anonymous snapshot: $OUT ($(find . -type f | wc -l) files, $(du -sh . | cut -f1))"
echo "Check it, zip it, upload the folder. Don't push it to a git remote that carries this repo's history."
