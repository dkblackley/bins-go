#!/bin/bash
# pacmann_setup.sh: build + install the native libs Pacmann's cgo deps need, without sudo.
#   NGT  (yahoojapan/gongt  -> NGT/Capi.h, libngt.so)
#   hnsw (evan176/hnswgo    -> libhnsw.so)
# Installs into the same places build.sh / the slurm scripts expect:
#   $HOME/opt/ngt-amd  and  $HOME/opt/hnsw
# Then writes $HOME/opt/pacmann_env.sh with the CGO flags; `source` it before `go build`.
#
# Needs: git, cmake, g++, make, go.   (Debian/Ubuntu: sudo apt install git cmake g++ make)
set -euo pipefail

NGT_PREFIX="${NGT_PREFIX:-$HOME/opt/ngt-amd}"
HNSW_PREFIX="${HNSW_PREFIX:-$HOME/opt/hnsw}"
BUILD_DIR="${BUILD_DIR:-$HOME/opt/src}"
JOBS="${JOBS:-$(nproc)}"
mkdir -p "$NGT_PREFIX" "$HNSW_PREFIX/lib" "$BUILD_DIR"

# ---------- NGT ----------
if [ ! -f "$NGT_PREFIX/lib/libngt.so" ]; then
  echo "==> Building NGT into $NGT_PREFIX"
  [ -d "$BUILD_DIR/NGT" ] || git clone --depth 1 https://github.com/yahoojapan/NGT.git "$BUILD_DIR/NGT"
  mkdir -p "$BUILD_DIR/NGT/build" && cd "$BUILD_DIR/NGT/build"
  # QBG needs BLAS/LAPACK and gongt doesn't use it, so turn it off to keep deps minimal
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$NGT_PREFIX" -DCMAKE_INSTALL_LIBDIR=lib -DNGT_QBG_DISABLED=ON
  make -j"$JOBS"
  make install
else
  echo "==> NGT already installed in $NGT_PREFIX"
fi
test -f "$NGT_PREFIX/include/NGT/Capi.h" || { echo "NGT/Capi.h missing after install"; exit 1; }

# ---------- hnsw (libhnsw.so from the hnswgo module source) ----------
if [ ! -f "$HNSW_PREFIX/lib/libhnsw.so" ]; then
  echo "==> Building libhnsw.so into $HNSW_PREFIX/lib"
  HNSWGO_VER=v0.0.0-20220622031020-39253a76f9e4   # the version pinned in bins-go/go.mod
  GOFLAGS=-mod=mod go mod download "github.com/evan176/hnswgo@$HNSWGO_VER"
  SRC="$(go env GOMODCACHE)/github.com/evan176/hnswgo@$HNSWGO_VER"
  rm -rf "$BUILD_DIR/hnswgo" && cp -r "$SRC" "$BUILD_DIR/hnswgo" && chmod -R u+w "$BUILD_DIR/hnswgo"
  make -C "$BUILD_DIR/hnswgo"
  cp "$BUILD_DIR/hnswgo/libhnsw.so" "$HNSW_PREFIX/lib/"
else
  echo "==> libhnsw.so already installed in $HNSW_PREFIX"
fi

# ---------- env file ----------
cat > "$HOME/opt/pacmann_env.sh" <<EOF
export NGT_PREFIX="$NGT_PREFIX"
export HNSW_PREFIX="$HNSW_PREFIX"
export CGO_CXXFLAGS="-std=c++11"
export CGO_CFLAGS="-I$NGT_PREFIX/include"
export CGO_LDFLAGS="-L$NGT_PREFIX/lib -Wl,-rpath,$NGT_PREFIX/lib -lngt -L$HNSW_PREFIX/lib -Wl,-rpath,$HNSW_PREFIX/lib -lhnsw"
EOF

echo
echo "Done. Before building bins-go run:"
echo "  source $HOME/opt/pacmann_env.sh && go build -o pir_app ./main.go"
