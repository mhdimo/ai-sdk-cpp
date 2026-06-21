# AI SDK for C++

A C++20 framework for building AI-powered applications. The orchestration layer between your application and LLM providers — inspired by the [Vercel AI SDK](https://sdk.vercel.ai), reimagined for native performance.

## Features

- **Multi-provider support** — Anthropic, OpenAI, Google, Amazon Bedrock, DeepSeek, MoonshotAI (plus z.ai/GLM via the OpenAI/Anthropic endpoints). See [docs/providers.md](docs/providers.md) for the full provider × protocol compatibility matrix.
- **Agentic tool loops** — Define tools with JSON schemas, let the model call them in a loop until task completion
- **Streaming** — SSE-based streaming with async generators
- **Structured output** — Generate validated JSON objects against a schema
- **Embeddings, images, speech, transcription, reranking** — Full multimodal support
- **MCP (Model Context Protocol)** — Connect to MCP servers for dynamic tool discovery
- **C++20 coroutines** — Native async/await with Boost.Asio
- **Language bindings** — C, Python, and Node.js bindings via a shared C FFI layer
- **Middleware** — Logging, caching, telemetry, and custom model middleware

## Quick Start

### Prerequisites

- C++20 compiler (Clang 15+, GCC 12+, MSVC 2022+)
- CMake 3.20+
- Boost 1.82+ (with JSON component)
- OpenSSL

### Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Basic Text Generation

```cpp
#include <ai/ai.hpp>
#include <ai/providers/anthropic/anthropic.hpp>
#include <boost/asio.hpp>
#include <iostream>

int main() {
    boost::asio::io_context ioc;

    auto anthropic = ai::providers::anthropic::create_anthropic({
        .io_context = ioc,
    });

    auto model = anthropic->language_model("claude-sonnet-4-20250514");

    auto task = ai::generate_text({
        .model = model,
        .prompt = "Explain C++20 coroutines in 3 sentences.",
        .max_output_tokens = 256,
        .temperature = 0.7,
    });

    task.start();
    while (!task.done()) {
        ioc.run_one();
    }

    auto result = task.get();
    std::cout << result.text << "\n";
    std::cout << "[Tokens: " << result.usage.input_tokens.total.value_or(0)
              << " in, " << result.usage.output_tokens.total.value_or(0) << " out]\n";
}
```

### Tool-Using Agent

```cpp
#include <ai/ai.hpp>
#include <ai/providers/anthropic/anthropic.hpp>
#include <boost/asio.hpp>
#include <iostream>

int main() {
    boost::asio::io_context ioc;

    auto anthropic = ai::providers::anthropic::create_anthropic({
        .io_context = ioc,
    });

    auto model = anthropic->language_model("claude-sonnet-4-20250514");

    ai::ToolSet tools;

    tools.add(ai::tool("get_weather",
        ai::schema::JsonSchema::object({
            {"city", ai::schema::JsonSchema::string("City name")}
        }).required({"city"}).additional_properties(false),
        "Get the current weather for a city",
        [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto city = std::string(input.at("city").as_string());
            boost::json::object weather;
            weather["city"] = city;
            weather["temperature"] = 22;
            weather["condition"] = "sunny";
            co_return boost::json::value(std::move(weather));
        }
    ));

    ai::ToolLoopAgent agent({
        .model = model,
        .tools = std::move(tools),
        .instructions = "You are a helpful assistant with access to a weather service.",
        .max_steps = 10,
    });

    auto task = agent.call("What's the weather in Paris?");

    task.start();
    while (!task.done()) {
        ioc.run_one();
    }

    auto result = task.get();
    std::cout << result.text << "\n";
}
```

### Streaming

```cpp
auto task = ai::stream_text({
    .model = model,
    .prompt = "Write a haiku about C++ templates.",
    .max_output_tokens = 128,
});
```

## Architecture

```
include/ai/         → Public headers (the API surface)
src/                → Core implementations
providers/          → One subdirectory per LLM provider
bindings/c/         → C shared library (universal FFI layer)
bindings/python/    → pybind11 wrapper over C API
bindings/node/      → N-API addon over C API
examples/           → Example applications
tests/              → Unit and integration tests
cli/                → CLI agent tool
```

### Key Abstractions

| Abstraction | Header | Purpose |
|---|---|---|
| `LanguageModel` | `include/ai/model/language_model.hpp` | Abstract base class for all providers |
| `ToolLoopAgent` | `include/ai/agent/tool_loop_agent.hpp` | Agentic loop: model → tools → repeat |
| `ToolDefinition` | `include/ai/tool/tool.hpp` | Tool = name + JSON schema + async function |
| `JsonSchema` | `include/ai/schema/json_schema.hpp` | Fluent schema builder + runtime validator |
| `Task<T>` | Core coroutine primitive | Single async result |
| `AsyncGenerator<T>` | `include/ai/stream/async_generator.hpp` | Yields values (for streaming) |

## Providers

### Full Implementations

These implement `LanguageModel` directly with custom prompt conversion and response parsing:

| Provider | Import |
|---|---|
| Anthropic | `#include <ai/providers/anthropic/anthropic.hpp>` |
| OpenAI | `#include <ai/providers/openai/openai.hpp>` |
| Google | `#include <ai/providers/google/google.hpp>` |
| Amazon Bedrock | `#include <ai/providers/bedrock/bedrock.hpp>` |

### OpenAI-Compatible Providers

Thin wrappers that delegate to the OpenAI provider with a different base URL and API key env var:

| Provider | Import |
|---|---|
| DeepSeek | `#include <ai/providers/deepseek/deepseek.hpp>` |
| MoonshotAI | `#include <ai/providers/moonshotai/moonshotai.hpp>` |

For any other OpenAI-compatible host (z.ai, self-hosted vLLM, etc.), use the generic OpenAI-compatible provider with a custom `base_url`: `#include <ai/providers/openai_compatible/openai_compatible.hpp>`. See [docs/providers.md](docs/providers.md) for details on z.ai/GLM, DeepSeek endpoints, and the compatibility matrix.

## Configuration

### CMake Options

| Option | Default | Description |
|---|---|---|
| `AI_SDK_BUILD_SHARED` | ON | Build shared library |
| `AI_SDK_BUILD_STATIC` | ON | Build static library |
| `AI_SDK_BUILD_TESTS` | ON | Build tests |
| `AI_SDK_BUILD_EXAMPLES` | ON | Build examples |
| `AI_SDK_BUILD_CLI` | ON | Build CLI agent tool |
| `AI_SDK_BUILD_BINDINGS` | ON | Build C bindings |
| `AI_SDK_BUILD_PYTHON` | OFF | Build Python pybind11 bindings |
| `AI_SDK_PROVIDER_<NAME>` | ON | Enable/disable individual providers |

### API Keys

Providers read API keys from environment variables:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export MOONSHOT_API_KEY="..."
```

## Testing

```bash
# Run all tests
ctest --test-dir build

# Run tests by tag
cd build && ./tests/unit/ai-sdk-tests "[sse]"
cd build && ./tests/unit/ai-sdk-tests "[schema]"
cd build && ./tests/unit/ai-sdk-tests "[message]"

# Run a single test by name
ctest --test-dir build -R "SseParser parses basic event"
```

Unit tests mock HTTP responses and run without network access. Integration tests require API keys.

## Language Bindings

### C API

The C binding (`bindings/c/ai_sdk.h`) exposes opaque handles with C calling convention:

```c
#include "ai_sdk.h"

ai_context_t ctx = ai_context_create();
ai_provider_t provider = ai_provider_create(ctx, "anthropic", NULL);
ai_model_t model = ai_model_create(provider, "claude-sonnet-4-20250514");
// ...
```

### Python

```bash
cmake .. -DAI_SDK_BUILD_PYTHON=ON
cmake --build .
```

```python
import ai_sdk

provider = ai_sdk.create_anthropic()
model = provider.language_model("claude-sonnet-4-20250514")
result = ai_sdk.generate_text(model=model, prompt="Hello!")
print(result.text)
```

### Node.js

The Node.js binding is a N-API addon wrapping the C API, available in `bindings/node/`.

## Examples

| Example | Description |
|---|---|
| `examples/basic_generate/` | Simple text generation |
| `examples/streaming_chat/` | Streaming response |
| `examples/tool_agent/` | Agent with calculator and weather tools |
| `examples/claude_code_clone/` | CLI coding agent |

## License

See [LICENSE](LICENSE) for details.
