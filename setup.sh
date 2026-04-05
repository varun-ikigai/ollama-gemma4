#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# Build patched Ollama from this fork (fixes Gemma 4 tool call parsing)
#
# Outputs:
#   ./dist/ollama              — the binary
#   ./dist/lib/ollama/runners/ — Metal/MLX runner libs
#
# After running this, from gemma-inference-server/:
#   ./serve.sh   (auto-detects ../ollama/dist/ollama)
#
# Prerequisites: Xcode CLI tools, Go 1.24+, cmake
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DIST_DIR="${SCRIPT_DIR}/dist"
ARCH="arm64"

echo ">> Checking prerequisites..."

if ! command -v go &>/dev/null; then
  echo "Error: Go is not installed. Install via: brew install go"
  exit 1
fi
echo "   Go: $(go version | awk '{print $3}')"

if ! command -v cmake &>/dev/null; then
  echo "   cmake not found — installing via Homebrew..."
  brew install cmake
fi
echo "   cmake: $(cmake --version | head -1)"

# ── Build native runners (Metal + MLX) ─────────────────────
echo ""
echo ">> Building native runners (Metal + MLX) for darwin/$ARCH..."

BUILD_DIR="${SCRIPT_DIR}/build/darwin-${ARCH}"
INSTALL_PREFIX="${SCRIPT_DIR}/dist/darwin-${ARCH}/"

cmake -B "$BUILD_DIR" \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DMLX_ENGINE=ON \
  -DOLLAMA_RUNNER_DIR=./

cmake --build "$BUILD_DIR" --target ggml-cpu -j
cmake --build "$BUILD_DIR" --target mlx mlxc -j
cmake --install "$BUILD_DIR" --component CPU
cmake --install "$BUILD_DIR" --component MLX

# ── Build Go binary ────────────────────────────────────────
echo ""
echo ">> Building ollama binary..."

VERSION=$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")

mkdir -p "$DIST_DIR"

CGO_ENABLED=1 \
GOARCH="${ARCH}" \
CGO_CFLAGS="-O3 -mmacosx-version-min=14.0" \
CGO_CXXFLAGS="-O3 -mmacosx-version-min=14.0" \
CGO_LDFLAGS="-mmacosx-version-min=14.0" \
go build -o "${DIST_DIR}/ollama" \
  -ldflags "-w -s -X github.com/ollama/ollama/version.Version=${VERSION}" \
  .

# ── Copy runner libs next to binary ────────────────────────
echo ">> Copying runner libraries..."
mkdir -p "${DIST_DIR}/lib/ollama/runners"
cp -R "${INSTALL_PREFIX}/lib/ollama/runners/"* "${DIST_DIR}/lib/ollama/runners/" 2>/dev/null || true

# ── Verify ──────────────────────────────────────────────────
echo ""
echo ">> Build complete!"
echo "   Binary: ${DIST_DIR}/ollama"
"${DIST_DIR}/ollama" --version
echo ""
echo ">> To use, from gemma-inference-server/:"
echo "   ./serve.sh"
