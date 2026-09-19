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

# Setting AI_SDK_LIBDIR is what tells binding.gyp this is a prebuild build, and
# that is the only thing it decides: with it set the addon records just the
# loader-relative rpath, and without it the in-repo build tree is added ahead of
# it for a source checkout. Nothing else needs to be passed down.

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
  # An unresolvable dependency reads `libai_sdk.so => not found`, which has no
  # path in it at all. Matched as though it were one, "not" lands in the
  # whitelist check below and comes back as `NOT RELOCATABLE: not` -- a message
  # that names neither the library nor the problem.
  ldd_out="$(ldd "$OUT/node.napi.node" "$OUT/$LIBNAME" 2>&1)"
  missing="$(printf '%s\n' "$ldd_out" | awk '/=>[[:space:]]+not found/ {print $1}')"
  if [ -n "$missing" ]; then
    echo "make-prebuild: these dependencies cannot be resolved, so the prebuild" >&2
    echo "               would not load on any machine but the one that built it:" >&2
    printf '    %s\n' $missing >&2
    exit 1
  fi
  deps="$(printf '%s\n' "$ldd_out" | awk '/=>/ {print $3}')"
fi
bad=0
while IFS= read -r dep; do
  [ -n "$dep" ] || continue
  case "$dep" in
    # A dependency that resolved to a file inside the package. That is $ORIGIN
    # doing its job, and it is what self-contained means rather than a
    # violation of it: ldd reports the path it *resolved*, so a correctly
    # rpath'd libai_sdk.so arrives here as an absolute path in $OUT. Without
    # this arm the check rejects a working Linux prebuild while passing the
    # macOS one, because otool -L prints the install name (@rpath/...) rather
    # than a resolved path -- so the same defect hides on the platform where
    # the check would have been read.
    "$OUT"/*) ;;
    # The system library directories as each platform's toolchain actually
    # spells them. Linux is the other trap: Ubuntu 22.04 prints system
    # libraries as /lib/<triple>/... , not /usr/lib/... . A whitelist written
    # from memory rejects every library on the machine, so this check fails on
    # a prebuild that is perfectly fine -- and, worse, it fails the same way on
    # one that is not, which is how it goes unread.
    /usr/lib/*|/lib/*|/lib64/*|/System/*|@rpath/*|@loader_path/*) ;;
    *) echo "    NOT RELOCATABLE: $dep" >&2; bad=1 ;;
  esac
done <<< "$deps"
if [ "$bad" -ne 0 ]; then
  echo "make-prebuild: the prebuild depends on a path outside the package." >&2
  echo "               Link that dependency statically instead." >&2
  exit 1
fi

# The rpath is asserted by its recorded value rather than inferred from the load
# test further down, because the load test cannot see this class of defect. A
# malformed entry does not fail to load -- it fails to *match*, and the loader
# moves on to the next entry. So an addon whose only relocatable rpath is inert
# still loads here, where the library is where the build put it, and breaks on a
# user's machine, which is the one place the check cannot run.
if [ "$PLATFORM" = "linux" ]; then
  runpath="$(objdump -p "$OUT/node.napi.node" | awk '/RUNPATH|RPATH/ {print $2}')"
  case "$runpath" in
    *'$ORIGIN'*) echo "    rpath: $runpath" ;;
    *)
      echo "make-prebuild: the addon's rpath is '${runpath:-<none>}', with no" >&2
      echo "               \$ORIGIN entry, so it will not find $LIBNAME beside it" >&2
      echo "               on a user's machine. See the OS=='linux' condition in" >&2
      echo "               binding.gyp." >&2
      exit 1
      ;;
  esac
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

# Two runtimes version their symbols here, and the one that actually decides
# whether the artifact starts is not the one named after the C library.
# `GLIBC_` and `GLIBCXX_` are unrelated namespaces -- a pattern for the first
# cannot match the second, whatever it is anchored to -- so reporting only the
# glibc floor answers a question nobody asked while the real bar goes unsaid.
# libstdc++ also moves faster than glibc does, by years, so an artifact can
# clear a comfortable glibc floor and still refuse to start.
#
# Stating both is all this does: the floor is set by the build image, not by
# this script, which is why the workflow matrix is named in the log.
if [ "$PLATFORM" = "linux" ]; then
  # The `|| true` is load-bearing on each of these. An absent namespace is a
  # *result*, not a failure -- and the one that goes absent is the one that says
  # the artifact is portable. A statically linked C++ runtime leaves no GLIBCXX_
  # references at all, so grep finds nothing, exits 1, pipefail carries that
  # through the pipeline, and `set -e` stops the script at the exact moment it
  # was about to report the good news. The run then ends in failure with no
  # message, which reads as a broken build rather than a portable one.
  syms="$( { objdump -T "$OUT/$LIBNAME"; objdump -T "$OUT/node.napi.node"; } 2>/dev/null || true )"
  glibc_floor="$(printf '%s\n' "$syms" | grep -o 'GLIBC_[0-9.]*' | sort -u -V | tail -1 || true)"
  cxx_floor="$(printf '%s\n' "$syms" | grep -o 'GLIBCXX_[0-9.]*' | sort -u -V | tail -1 || true)"
  echo "    glibc floor:     ${glibc_floor:-unknown}"
  echo "    libstdc++ floor: ${cxx_floor:-unknown}"
  echo "    (both set by the build image -- see the workflow matrix)"
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
