#!/usr/bin/env bash
# Run the gated live smoke test against a real provider.
#
# Usage:
#   ANTHROPIC_API_KEY=sk-... ./scripts/run_smoke.sh
#
# Optional env:
#   ANTHROPIC_MODEL       (default: claude-sonnet-4-5)
#   ANTHROPIC_BASE_URL    (default: https://api.anthropic.com)
#   AI_SDK_RUN_LIVE=1     (required to actually run; this script sets it)
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY is not set; cannot run the live smoke test." >&2
  exit 1
fi

export AI_SDK_RUN_LIVE=1

cmake --build build -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" >/dev/null
exec ./build/tests/unit/ai-sdk-live-tests --verbosity rich
