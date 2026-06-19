# Contributing to ai-sdk-cpp

Thanks for your interest in improving ai-sdk-cpp! This is a short guide to get
you productive.

## Building & testing

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(sysctl -n hw.ncpu)"        # macOS; use $(nproc) on Linux
ctest --test-dir build                                # full suite
cd build && ./tests/unit/ai-sdk-tests "[streaming]"   # by tag
```

Providers can be toggled to speed up builds
(`-DAI_SDK_PROVIDER_BEDROCK=OFF -DAI_SDK_PROVIDER_GOOGLE=OFF`).

## Code style

- **C++20**, coroutines (`Task<T>` / `AsyncGenerator<T>`), Boost.Asio event loop.
- Format with `clang-format` (config committed). Aim for `clang-tidy` clean on
  new code (config committed; it is intentionally conservative).
- Match the surrounding code: `namespace ai::`, `#pragma once`, defensive JSON
  parsing (`.find()` + type checks, never `.at()` on provider responses), 4-space
  indent, attached braces.
- Public API in `include/ai/`; implementations in `src/`, `providers/`, etc.
  Re-export new public headers from `include/ai/ai.hpp`.

## Tests

Every new feature or bug fix ships with Catch2 v3 unit tests under
`tests/unit/` (tagged, e.g. `[session]`). Provider-facing code is tested with
the in-memory `FakeHttpClient` (see `tests/unit/fake_http_client.hpp`) so the
suite runs fully offline. **Do not add tests that require network access by
default** — gate live tests behind an env var (see
`tests/unit/test_live_smoke.cpp`).

## Adding a provider

See `CLAUDE.md` ("Adding a New OpenAI-Compatible Provider"). Full providers
implement `LanguageModel` directly; thin wrappers delegate to the OpenAI
provider with a different `base_url` + env var.

## Pull requests

1. Branch from `main`.
2. Keep changes focused; one logical change per PR.
3. Ensure `ctest --test-dir build` is green.
4. Describe the *why* in the PR description; link any issue.

## Reporting issues / security

See [SECURITY.md](SECURITY.md) for responsible disclosure of vulnerabilities.
