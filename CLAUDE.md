# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Full build (from repo root)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(sysctl -n hw.ncpu)

# Rebuild after changes (from build/)
cmake --build . -j$(sysctl -n hw.ncpu)

# Run all tests
ctest --test-dir build

# Run a single test by name
ctest --test-dir build -R "SseParser parses basic event"

# Run tests matching a tag
cd build && ./tests/unit/ai-sdk-tests "[sse]"
cd build && ./tests/unit/ai-sdk-tests "[schema]"
cd build && ./tests/unit/ai-sdk-tests "[message]"

# Build with Python bindings
cmake .. -DAI_SDK_BUILD_PYTHON=ON

# Disable specific providers to speed up builds
cmake .. -DAI_SDK_PROVIDER_BEDROCK=OFF -DAI_SDK_PROVIDER_GOOGLE=OFF
```

## Architecture

This is a C++20 AI agent framework — the orchestration layer between your application and LLM providers. Think of it as the Vercel AI SDK rewritten in C++ with language bindings for Python, TypeScript, Rust, Go, and C.

### Layered Design

```
include/ai/         → Public headers (the API surface)
src/                → Core implementations
providers/          → One subdirectory per LLM provider
bindings/c/         → C shared library (universal FFI layer)
bindings/python/    → pybind11 wrapper over C API
bindings/node/      → N-API addon over C API
```

### Key Abstractions

- **`LanguageModel`** (`include/ai/model/language_model.hpp`) — Abstract base class. Every provider implements `do_generate()` and `do_stream()` returning `Task<T>`.
- **`ToolLoopAgent`** (`include/ai/agent/tool_loop_agent.hpp`) — The agent loop. Calls the model, executes tools, appends results, repeats until `max_steps` or finish.
- **`ToolDefinition`** (`include/ai/tool/tool.hpp`) — Tool = name + JSON schema + async execute function. The `tool()` factory creates them.
- **`Task<T>`** / **`AsyncGenerator<T>`** (`include/ai/stream/async_generator.hpp`) — Coroutine primitives. Task is a single async result, AsyncGenerator yields values.

### Provider Pattern

**Full providers** (Anthropic, OpenAI, Google, Bedrock) implement `LanguageModel` directly with custom prompt conversion and response parsing. The conversion and parsing logic lives **inline in the provider's `*_model.cpp`** (`build_request_body()` builds the request body, `parse_response()` parses the reply); `convert_prompt.cpp` and `parse_response.cpp` are currently empty placeholders reserved for future extraction.
```
providers/anthropic/
├── include/ai/providers/anthropic/
│   ├── anthropic.hpp          ← create_anthropic() factory
│   └── anthropic_model.hpp    ← LanguageModel implementation
└── src/
    ├── anthropic_model.cpp    ← build_request_body() + parse_response() (the real logic)
    ├── convert_prompt.cpp     ← placeholder (reserved for future extraction)
    └── parse_response.cpp     ← placeholder (reserved for future extraction)
```

**OpenAI-compatible providers** (DeepSeek, Groq, xAI, Mistral, Fireworks, TogetherAI, Perplexity, Cohere, MoonshotAI) are thin wrappers that delegate to the OpenAI provider with a different base URL:
```
providers/deepseek/
├── include/ai/providers/deepseek/deepseek.hpp
└── src/deepseek_provider.cpp   ← ~30 lines, sets base_url + env var name
```

### C Binding as FFI Bridge

`bindings/c/ai_sdk.h` exposes opaque handles (`ai_context_t`, `ai_provider_t`, `ai_model_t`, `ai_agent_t`, `ai_tool_set_t`) with C calling convention. All language bindings (Python, Node, Rust, Go) wrap this single C API. The implementation is in `bindings/c/ai_sdk.cpp`.

### Async Model

The SDK uses C++20 coroutines with Boost.Asio `io_context` as the event loop. For the C API and bindings, blocking wrappers drive the coroutines internally so callers don't need to manage an event loop.

## Coding Conventions

- **Defensive JSON parsing**: Always use `.find()` + type checks, never `.at()` on provider responses. Providers send malformed/partial JSON regularly.
- **Error hierarchy**: Errors in `include/ai/error/` — `ApiCallError` is the base, specialized into `RateLimitError`, `AuthenticationError`, `TimeoutError`, etc. The HTTP layer auto-maps status codes to error types.
- **Schema system**: `ai::schema::JsonSchema` provides a fluent builder (`.object({...}).required({}).additional_properties(false)`) and a runtime `Validator`.
- **Namespaces**: `ai::` for core, `ai::providers::anthropic::` etc. for providers, `ai::schema::` for schema, `ai::stream::` for streaming primitives.
- **Test framework**: Catch2 v3. Tests are in `tests/unit/`, discovered via `catch_discover_tests()`. Each TEST_CASE gets a tag like `[sse]`, `[schema]`, `[message]`.

## Adding a New OpenAI-Compatible Provider

1. Create `providers/<name>/` with the standard structure
2. The provider .cpp creates an OpenAI model instance with a custom `base_url` and reads API key from a provider-specific env var
3. Add `option(AI_SDK_PROVIDER_<NAME> ...)` and `add_subdirectory(providers/<name>)` to root CMakeLists.txt
4. Add the library to `bindings/c/CMakeLists.txt` target_link_libraries
5. Add a case in `bindings/c/ai_sdk.cpp` `ai_provider_create()` switch

## Testing Against Live APIs

Integration tests require API keys set as environment variables. The unit tests mock HTTP responses and run without network access.
