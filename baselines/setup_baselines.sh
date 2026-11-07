#!/bin/bash
# One-off setup for the Tiptoe and Coeus baselines. Run on a login node (needs internet + a C++ compiler).
#
#   ./baselines/setup_baselines.sh sim      # only ./simplepir_bench, for the simulated Tiptoe runs (no SEAL)
#   ./baselines/setup_baselines.sh tiptoe   # SEAL 4.1.1 + pir_app_tiptoe (pir_app with -tags tiptoe)
#   ./baselines/setup_baselines.sh coeus    # SEAL 3.5.6 + Coeus' matmult/bin/opt2 benchmark
#   ./baselines/setup_baselines.sh all
#
# Everything goes under $OPT (default ~/opt) and needs no root.
#
# Compiler: Tiptoe's underhood/rlwe/rlwe.go hard-codes `-std=c++20 -Werror -march=native`, so it needs GCC >= 10.
# On Hopper that means a newer module than the gnu9/9.3.0 used for the main build, e.g. `module load gnu12`
# (check `module avail gnu`). -march=native means: build on the same CPU type you run on (the slurm uses --constraint=amd).

set -euo pipefail
cd "$(dirname "$0")/.."

OPT="${OPT:-$HOME/opt}"
JOBS="${JOBS:-8}"
WHAT="${1:-all}"
mkdir -p "$OPT/src"

build_seal() {   # $1 = version tag, $2 = install prefix, rest = extra cmake flags
    local ver=$1 prefix=$2; shift 2
    if [[ -d "$prefix/lib" || -d "$prefix/lib64" ]]; then echo "SEAL $ver already in $prefix"; return; fi
    local src="$OPT/src/SEAL-$ver"
    [[ -d "$src" ]] || git clone --depth 1 -b "v$ver" https://github.com/microsoft/SEAL "$src"
    cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
        -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF -DSEAL_USE_ZSTD=OFF "$@"
    cmake --build "$src/build" -j "$JOBS"
    cmake --install "$src/build"
}

seal41_env() {
    local p="$OPT/seal-4.1"
    local lib="$p/lib"; [[ -d "$p/lib64" ]] && lib="$p/lib64"
    export CGO_CXXFLAGS="-I$p/include/SEAL-4.1 ${CGO_CXXFLAGS:-}"
    export CGO_LDFLAGS="-L$lib -Wl,-rpath,$lib ${CGO_LDFLAGS:-}"
}

if [[ "$WHAT" == "sim" || "$WHAT" == "all" ]]; then
    go build -o simplepir_bench ./cmd/simplepir_bench
    echo "simplepir_bench built"
fi

if [[ "$WHAT" == "tiptoe" || "$WHAT" == "all" ]]; then
    gcc_major=$(g++ -dumpversion | cut -d. -f1)
    if (( gcc_major < 10 )); then
        echo "g++ $gcc_major is too old for Tiptoe's -std=c++20 (need >= 10); load a newer gnu module first" >&2
        exit 1
    fi
    # Tiptoe's README: SEAL must be built with SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=off
    build_seal 4.1.1 "$OPT/seal-4.1" -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF -DBUILD_SHARED_LIBS=ON

    # The main build's NGT/HNSW flags are still needed: pir_app_tiptoe is the normal binary plus Tiptoe.
    export NGT_PREFIX="${NGT_PREFIX:-$HOME/opt/ngt-amd}"
    export HNSW_PREFIX="${HNSW_PREFIX:-$HOME/opt/hnsw}"
    export CGO_CFLAGS="-I${NGT_PREFIX}/include ${CGO_CFLAGS:-}"
    export CGO_LDFLAGS="-L${NGT_PREFIX}/lib -Wl,-rpath,${NGT_PREFIX}/lib -lngt -L${HNSW_PREFIX}/lib -Wl,-rpath,${HNSW_PREFIX}/lib -lhnsw"
    seal41_env
    echo "Building pir_app_tiptoe ..."
    go build -tags tiptoe -o pir_app_tiptoe .
    echo "pir_app_tiptoe built (run with -t tiptoe)"
fi

if [[ "$WHAT" == "coeus" || "$WHAT" == "all" ]]; then
    # matmult/CMakeLists.txt asks for find_package(SEAL 3.5) and uses the 3.5 API (SEALContext::Create)
    build_seal 3.5.6 "$OPT/seal-3.5" -DBUILD_SHARED_LIBS=ON -DSEAL_BUILD_DEPS=ON
    src="$OPT/src/Coeus_artifact"
    [[ -d "$src" ]] || git clone --depth 1 https://github.com/ishtiyaque/Coeus_artifact "$src"
    cmake -S "$src/matmult" -B "$src/matmult/build" -DCMAKE_PREFIX_PATH="$OPT/seal-3.5" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$src/matmult/build" -j "$JOBS"
    echo "Coeus benchmark: $src/matmult/bin/opt2   (pass it as --opt2-bin to baselines/coeus/run_coeus.py)"
fi
