#!/bin/bash
# Shared Go + cgo environment for building pir_app: `source go_env.sh` (build.sh, run_tree.slurm).
# Override anything from the calling shell: GOROOT=..., NGT_PREFIX=..., HNSW_PREFIX=...

module --quiet purge 2>/dev/null || true
module load hosts/hopper gnu9/9.3.0 2>/dev/null || true

# Native libs for pacmann (built by setup_pacmann.sh)
export NGT_PREFIX="${NGT_PREFIX:-$HOME/opt/ngt-amd}"
export HNSW_PREFIX="${HNSW_PREFIX:-$HOME/opt/hnsw}"
export CGO_CXXFLAGS="-std=c++11"
export CGO_CFLAGS="-I${NGT_PREFIX}/include"
export CGO_LDFLAGS="-L${NGT_PREFIX}/lib -Wl,-rpath,${NGT_PREFIX}/lib -lngt -L${HNSW_PREFIX}/lib -Wl,-rpath,${HNSW_PREFIX}/lib -lhnsw"
for lib in "$NGT_PREFIX/lib/libngt.so" "$HNSW_PREFIX/lib/libhnsw.so"; do
    [[ -f "$lib" ]] || echo "go_env.sh: WARNING $lib missing, run setup_pacmann.sh (or set NGT_PREFIX/HNSW_PREFIX)" >&2
done

# Go >= 1.24 (go.mod): $GOROOT, else a cached 1.24+ toolchain, else /usr/local/go, else `go` on PATH
go_ok() { [[ -x "$1/bin/go" ]] && "$1/bin/go" version 2>/dev/null | grep -qE 'go1\.(2[4-9]|[3-9][0-9])'; }
if ! go_ok "${GOROOT:-/nonexistent}"; then
    GOROOT=""
    for c in $(ls -d "$HOME"/go/pkg/mod/golang.org/toolchain@*go1.2[4-9]*.linux-amd64 2>/dev/null | sort -rV) \
             /usr/local/go "$(command -v go >/dev/null 2>&1 && go env GOROOT 2>/dev/null)"; do
        [[ -n "$c" ]] && go_ok "$c" && { GOROOT="$c"; break; }
    done
fi
if [[ -z "$GOROOT" ]]; then
    echo "go_env.sh: no Go >= 1.24 found; set GOROOT=<dir with bin/go>" >&2
    exit 1
fi
export GOROOT PATH="$GOROOT/bin:$PATH" GOTOOLCHAIN=local
echo "go_env.sh: $(go version) GOROOT=$GOROOT NGT=$NGT_PREFIX HNSW=$HNSW_PREFIX" >&2
