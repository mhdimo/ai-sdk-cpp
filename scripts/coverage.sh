#!/usr/bin/env bash
# What do the tests actually exercise?
#
# Clang source-based coverage: builds instrumented, runs ctest, merges the
# profiles, reports per file, then names the source files no test touched.
#
# Scope: the OFFLINE C++ SUITE only. The Node binding suite is not part of this
# profile, and that leaves a real hole — the C++ tests substitute their own
# ai::http::IHttpClient, so src/http/* and the C binding are only reached
# incidentally. The Node suite drives both for real. Read the two together.
#
# The instrumented build goes to its OWN tree (build-coverage/), never the
# default build/. That matters: the Node addon links build/bindings/c at
# runtime, so instrumenting that directory in place would slow every Node test
# down and litter .profraw files through it.
#
# Usage:
#   ./scripts/coverage.sh            # headline + full report
#   ./scripts/coverage.sh --html     # ... and browsable HTML
#
# Env:
#   AI_SDK_COVERAGE_BUILD_DIR   build tree to use (default: build-coverage)
#   AI_SDK_COVERAGE_CMAKE_ARGS  extra cmake args, e.g. to disable a slow provider
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR="${AI_SDK_COVERAGE_BUILD_DIR:-build-coverage}"
COV_DIR="$BUILD_DIR/coverage"
PROF_DIR="$COV_DIR/profraw"
MERGED="$COV_DIR/merged.profdata"
HTML=0

for arg in "$@"; do
  case "$arg" in
    --html) HTML=1 ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# The profile format is version-tied to the compiler, so prefer the toolchain
# that ships with it — on macOS that is the Xcode one, not some Homebrew LLVM
# that happens to be on PATH.
find_llvm_tool() {
  local name="$1"
  if [[ "$(uname)" == "Darwin" ]] && xcrun --find "$name" >/dev/null 2>&1; then
    xcrun --find "$name"; return 0
  fi
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"; return 0
  fi
  local p
  for p in /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin /usr/lib/llvm-*/bin; do
    if [[ -x "$p/$name" ]]; then echo "$p/$name"; return 0; fi
  done
  echo "error: $name not found. Install LLVM (macOS: xcode-select --install, or brew install llvm)." >&2
  return 1
}

PROFDATA="$(find_llvm_tool llvm-profdata)"
LLVM_COV="$(find_llvm_tool llvm-cov)"
JOBS="$( (sysctl -n hw.ncpu 2>/dev/null || nproc) )"

echo "==> Configuring $BUILD_DIR with AI_SDK_COVERAGE=ON"
# Compile each source file into exactly ONE image. The default build makes both a
# static and a shared core from the same sources under the same output name, so
# every core function would appear twice and llvm-cov would double-count it.
# BUILD_SHARED_LIBS=ON pulls the providers and the C binding into shared objects
# too, which is also what makes their coverage visible at all — see the note on
# OBJECTS below.
COVERAGE_CMAKE_ARGS="-DAI_SDK_BUILD_STATIC=OFF -DBUILD_SHARED_LIBS=ON"
# shellcheck disable=SC2086  # both the defaults and the override are split on purpose
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DAI_SDK_COVERAGE=ON \
  $COVERAGE_CMAKE_ARGS ${AI_SDK_COVERAGE_CMAKE_ARGS:-} >/dev/null

echo "==> Building"
cmake --build "$BUILD_DIR" -j"$JOBS" >/dev/null

rm -rf "$COV_DIR"
mkdir -p "$PROF_DIR"

# %p keeps concurrent test processes from clobbering each other; %m separates
# the profile written by each linked image (the test binary and every dylib).
echo "==> Running ctest"
LLVM_PROFILE_FILE="$PWD/$PROF_DIR/%p-%m.profraw" \
  ctest --test-dir "$BUILD_DIR" --output-on-failure

if ! compgen -G "$PROF_DIR/*.profraw" >/dev/null; then
  echo "error: no .profraw files were written — was the build instrumented?" >&2
  exit 1
fi

echo "==> Merging profiles"
"$PROFDATA" merge -sparse "$PROF_DIR"/*.profraw -o "$MERGED"

# Every instrumented image has to be named, or the coverage recorded inside it is
# silently dropped. Two traps here, both of which produce a plausible-but-wrong
# number rather than an error:
#
#   1. Each image must be passed with its own -object= flag. Passing them as
#      positional arguments looks like it works, but llvm-cov uses only the FIRST
#      one and ignores the rest, so you silently get one library's coverage
#      reported as the project's.
#   2. A static archive is not enough. ld64 does not carry the __LLVM_COV
#      sections of archive members into the linked binary, so a library that is
#      only ever linked statically has to be passed as a .a and read directly.
#      The coverage build avoids the situation by making everything shared.
OBJECTS=()
add_object() { OBJECTS+=("-object=$1"); }
while IFS= read -r -d '' f; do add_object "$f"; done < <(
  find "$BUILD_DIR" -type f \( -name '*.dylib' -o -name '*.so' \) -not -path '*/_deps/*' -print0
)
# The test binary carries the tests plus any header-only code they instantiate.
if [[ -x "$BUILD_DIR/tests/unit/ai-sdk-tests" ]]; then
  add_object "$BUILD_DIR/tests/unit/ai-sdk-tests"
fi

if [[ ${#OBJECTS[@]} -eq 0 ]]; then
  echo "error: no instrumented binaries found under $BUILD_DIR" >&2
  exit 1
fi

# Everything that is not this project's library code.
EXTERNAL='(^|/)(_deps|build|build-coverage)/|/usr/|/opt/homebrew/|/Applications/Xcode|/Library/Developer/'
# The shipped library: core, each provider, and the C binding.
LIBRARY="$EXTERNAL|(^|/)(tests|cli|examples)/"
# Public headers show up as one-line rows (a default argument, an inline
# accessor). They are real coverage but they drown the table.
IMPL="$LIBRARY|\.(h|hpp|ipp|inl)$"

# llvm-cov notes "N functions have mismatched data" whenever the same header-only
# function is instantiated in more than one image, which is the normal case here.
# Collect those instead of letting them land in the middle of a table.
LLVM_NOTES="$COV_DIR/llvm-cov.log"
: > "$LLVM_NOTES"

report() {
  "$LLVM_COV" report "${OBJECTS[@]}" \
    -instr-profile="$MERGED" \
    -ignore-filename-regex="$1" 2>>"$LLVM_NOTES"
}

echo
echo "============= library implementation, exercised by the offline tests ============="
report "$IMPL"

echo
echo "Note: src/http/* and bindings/c/* look untouched above because the C++ tests"
echo "mock at ai::http::IHttpClient, one layer above them. The Node suite exercises"
echo "both against a real socket; it is not included in this profile."

echo
echo "============== the same, with the public headers folded in =============="
# Headers only: the CLI and example binaries are never run by ctest, so they have
# no profile to contribute and passing them would abort llvm-cov.
report "$LIBRARY" | tail -1

# The actionable half of the report: what did no test touch at all?
echo
echo "=========================== never executed by any test ==========================="
BODY="$(report "$IMPL" | sed -n '3,$p')"
TOTAL_FILES="$(echo "$BODY" | awk 'NF>1 && $2>0' | wc -l | tr -d ' ')"
UNTOUCHED="$(echo "$BODY" | awk 'NF>1 && $8>0 && $8==$9 {print $1}' | sort)"
if [[ -z "$UNTOUCHED" ]]; then
  echo "  (none — every source file ran at least once)"
else
  echo "$UNTOUCHED" | sed 's/^/  /'
  echo
  echo "  $(echo "$UNTOUCHED" | wc -l | tr -d ' ') of $TOTAL_FILES source files"
fi

if [[ "$HTML" == 1 ]]; then
  echo
  echo "==> Writing HTML to $COV_DIR/html"
  "$LLVM_COV" show "${OBJECTS[@]}" \
    -instr-profile="$MERGED" \
    -ignore-filename-regex="$EXTERNAL" \
    -format=html -output-dir="$COV_DIR/html" >/dev/null 2>>"$LLVM_NOTES"
  echo "    open $COV_DIR/html/index.html"
fi

if [[ -s "$LLVM_NOTES" ]]; then
  echo
  echo "==> llvm-cov notes"
  sed 's/\x1b\[[0-9;]*m//g' "$LLVM_NOTES" | sort -u | sed 's/^/    /'
fi
