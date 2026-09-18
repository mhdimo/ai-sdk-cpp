#!/usr/bin/env bash
#
# Build a relocatable Node prebuild for the current platform into
# bindings/node/prebuilds/<platform>-<arch>/.
#
# Run from anywhere; the repo is found relative to this script. On CI this is
# the whole per-platform story: the workflow runs it once per runner and
# uploads the directory.
#
# Two properties make the result relocatable, and both are load-bearing:
#
#   1. libai_sdk is linked against a *static* OpenSSL. A dynamic one would
#      either record the build machine's Homebrew or distro path -- unloadable
#      on anyone else's machine -- or, if bundled, put a second libssl into a
#      Node process that already has its own. Node links OpenSSL statically,
#      so a bundled dynamic copy is a well known source of symbol conflicts.
#
#   2. The addon is linked with -Wl,-rpath,@loader_path, so the loader looks
#      for the shared library in the same directory as the .node file, wherever
#      the package manager happened to put it.
#
# The output is one self-contained directory: node.napi.node with the shared
# library beside it. node-gyp-build finds the first by platform tuple; the
# dynamic loader finds the second by rpath.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NODE_DIR="$ROOT/bindings/node"
BUILD_DIR="${AI_SDK_PREBUILD_BUILD_DIR:-$ROOT/build-prebuild}"

# process.stdout.write everywhere rather than `node -p`: -p formats through
# util.inspect, which quotes strings and may colourise when stdout is a tty.
# Either would corrupt the tuple and the -j argument.
PLATFORM="$(node -e 'process.stdout.write(process.platform)')"
ARCH="$(node -e 'process.stdout.write(process.arch)')"
TUPLE="$PLATFORM-$ARCH"
JOBS="$(node -e 'process.stdout.write(String(require("os").cpus().length))')"

# Must match MACOSX_DEPLOYMENT_TARGET in binding.gyp. If they differ, the dylib
# records the build machine's macOS as its minimum and the prebuild advertises
# a floor it cannot honour -- it links fine here and fails on the user's older
# machine, which is the one place the failure is expensive.
MIN_MACOS="12.0"

case "$PLATFORM" in
  darwin) LIBNAME="libai_sdk.dylib" ;;
  linux)  LIBNAME="libai_sdk.so" ;;
  *)
    echo "make-prebuild: no prebuild recipe for platform '$PLATFORM'" >&2
    exit 1
    ;;
esac

echo "==> $TUPLE: building libai_sdk with a static OpenSSL"
cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENSSL_USE_STATIC_LIBS=TRUE \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$MIN_MACOS" \
  -DAI_SDK_BUILD_TESTS=OFF \
  -DAI_SDK_BUILD_EXAMPLES=OFF \
  -DAI_SDK_BUILD_CLI=OFF \
  -DAI_SDK_BUILD_PYTHON=OFF \
  >/dev/null
cmake --build "$BUILD_DIR" --target ai-sdk-c -j"$JOBS" >/dev/null

LIBDIR="$BUILD_DIR/bindings/c"
[ -f "$LIBDIR/$LIBNAME" ] || { echo "make-prebuild: $LIBDIR/$LIBNAME missing" >&2; exit 1; }

echo "==> $TUPLE: building the addon against it"
(cd "$NODE_DIR" && AI_SDK_LIBDIR="$LIBDIR" npx node-gyp rebuild >/dev/null)

echo "==> $TUPLE: compiling the JavaScript"
(cd "$NODE_DIR" && npx tsc)

OUT="$NODE_DIR/prebuilds/$TUPLE"
rm -rf "$OUT"
mkdir -p "$OUT"
cp "$NODE_DIR/build/Release/ai_sdk_native.node" "$OUT/node.napi.node"
cp "$LIBDIR/$LIBNAME" "$OUT/$LIBNAME"

# A prebuild that depends on anything outside the platform's system libraries
# is broken for everyone but the machine that built it -- which is exactly how
# the Homebrew OpenSSL path went unnoticed until the package was packed.
echo "==> $TUPLE: checking the result is self-contained"
if [ "$PLATFORM" = "darwin" ]; then
  deps="$(otool -L "$OUT/node.napi.node" "$OUT/$LIBNAME" | awk '/^\t/ {print $1}')"
else
  deps="$(ldd "$OUT/node.napi.node" "$OUT/$LIBNAME" | awk '/=>/ {print $3} $2 == "=>" {print $3}')"
fi
bad=0
while IFS= read -r dep; do
  [ -n "$dep" ] || continue
  case "$dep" in
    /usr/lib/*|/System/*|@rpath/*|@loader_path/*) ;;
    *) echo "    NOT RELOCATABLE: $dep" >&2; bad=1 ;;
  esac
done <<< "$deps"
if [ "$bad" -ne 0 ]; then
  echo "make-prebuild: the prebuild depends on a path outside the package." >&2
  echo "               Link that dependency statically instead." >&2
  exit 1
fi

# Both binaries must record the same minimum macOS, or the prebuild advertises
# a floor it cannot honour: it links and loads here, then fails on the older
# machine. Nothing else in this script can catch that, because the build host
# is always new enough to run it.
if [ "$PLATFORM" = "darwin" ]; then
  for f in "$OUT/node.napi.node" "$OUT/$LIBNAME"; do
    minos="$(otool -l "$f" | awk '/LC_BUILD_VERSION/{found=1} found && /minos/{print $2; exit}')"
    [ -n "$minos" ] || { echo "make-prebuild: no LC_BUILD_VERSION in $f" >&2; exit 1; }
    if awk -v a="$minos" -v b="$MIN_MACOS" 'BEGIN{
          split(a, x, "."); split(b, y, ".")
          exit !(x[1]+0 > y[1]+0 || (x[1]+0 == y[1]+0 && x[2]+0 > y[2]+0))
        }'; then
      echo "    $(basename "$f") requires macOS $minos, above the $MIN_MACOS target" >&2
      exit 1
    fi
    echo "    $(basename "$f") minos $minos"
  done
fi

# glibc versions its symbols, so a binary built against a newer glibc refuses to
# start on an older one. That makes the build image -- not this script -- decide
# what the artifact runs on, which is worth stating out loud in the log rather
# than leaving for someone to discover from a user's bug report.
if [ "$PLATFORM" = "linux" ]; then
  floor="$( { objdump -T "$OUT/$LIBNAME"; objdump -T "$OUT/node.napi.node"; } 2>/dev/null |
            grep -o 'GLIBC_[0-9.]*' | sort -u -V | tail -1 )"
  echo "    glibc floor: ${floor:-unknown} (set by the build image, see the workflow matrix)"
fi

# Load the prebuild the way a user's install would, with PREBUILDS_ONLY so the
# in-repo build/Release is not what answers. This is the only check that
# actually exercises the shipped layout: node-gyp-build's tuple lookup, the
# rpath, and the shared library all have to line up for it to pass.
echo "==> $TUPLE: loading it the way an install would"
(cd "$NODE_DIR" && PREBUILDS_ONLY=1 node -e "
  const ai = require('.');
  console.log('    loaded, version ' + ai.version());
  // The addon holds a Context open, so exit explicitly rather than waiting
  // for a loop that will never drain.
  process.exit(0);
")

echo "==> $TUPLE: wrote $OUT"
ls -la "$OUT"

# The addon was built in prebuild mode, which is to say linked with only
# @loader_path -- correct for the shipped directory, wrong for build/Release,
# where the dylib is not beside the addon but in build/bindings/c. That build
# overwrote whatever was there, so put the source-tree addon back. Skipped when
# there is no in-repo library to link against, which is the case on a CI runner
# that only builds prebuilds.
DEV_LIBDIR="$ROOT/build/bindings/c"
if [ -d "$DEV_LIBDIR" ]; then
  echo "==> $TUPLE: restoring the source-tree addon"
  (cd "$NODE_DIR" && npx node-gyp rebuild >/dev/null)
else
  echo "==> $TUPLE: no $DEV_LIBDIR, leaving build/Release as the prebuild"
fi
