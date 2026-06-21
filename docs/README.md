# ai-sdk-cpp

A high-performance C++20 AI agent framework.

Like **llama.cpp** is to inference, **ai-sdk-cpp** is to agent orchestration. One native engine, any language.

- **6 providers** — Anthropic, OpenAI, Google, Amazon Bedrock, DeepSeek, MoonshotAI (plus z.ai/GLM via the OpenAI/Anthropic endpoints). See [providers.md](providers.md) for the full compatibility matrix.
- **Automatic tool loops** — define tools, the agent calls them until the task is done
- **Streaming** — SSE parsing, token-by-token output, backpressure
- **Multi-language** — C++, Python, TypeScript/Node.js, and C — or any language via the shared C FFI
- **C++20 coroutines** — native async I/O, no threads for simple use cases

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Your Agent (C++ / Python / TypeScript / C)             │
├─────────────────────────────────────────────────────────┤
│  Bindings  (pybind11 / N-API / C header / FFI)          │
├─────────────────────────────────────────────────────────┤
│  Core      (generate_text · stream_text · Agent loop)   │
├─────────┬────────┬────────┬────────┬────────┬───────────┤
│Anthropic│ OpenAI │ Google │ Bedrock│DeepSeek│ MoonshotAI│
├─────────┴────────┴────────┴────────┴────────┴───────────┤
│  HTTP/TLS (Boost.Beast) + SSE Parser + WebSocket        │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### C++

```cpp
#include <ai/ai.hpp>
#include <ai/providers/anthropic/anthropic.hpp>
#include <boost/asio.hpp>
#include <iostream>

int main() {
    boost::asio::io_context ioc;
    auto anthropic = ai::providers::anthropic::create_anthropic({.io_context = ioc});
    auto model = anthropic->language_model("claude-sonnet-4-20250514");

    auto task = ai::generate_text({.model = model, .prompt = "Hello!"});
    task.start();
    while (!task.done()) ioc.run_one();

    std::cout << task.get().text << "\n";
}
```

### Python

```python
from ai_sdk import create_anthropic, generate_text

model = create_anthropic()("claude-sonnet-4-20250514")
result = generate_text(model, prompt="Hello!")
print(result.text)
```

### TypeScript

```typescript
import { createAnthropic, generateText } from 'ai-sdk-cpp';

const model = createAnthropic()('claude-sonnet-4-20250514');
const { text } = await generateText({ model, prompt: 'Hello!' });
console.log(text);
```

### C

```c
#include "ai_sdk.h"
#include <stdio.h>

int main(void) {
    ai_context_t ctx = ai_context_create();
    ai_provider_options_t opts = { .api_key = NULL, .base_url = NULL };
    ai_provider_t provider = ai_provider_create(ctx, "anthropic", opts);
    ai_model_t model = ai_model_create(provider, "claude-sonnet-4-20250514");

    ai_generate_options_t gen = { .model = model, .prompt = "Hello!" };
    ai_generate_result_t result;
    ai_generate_text(gen, &result);

    printf("%s\n", result.text);
    ai_generate_result_free(&result);
    ai_model_destroy(model);
    ai_provider_destroy(provider);
    ai_context_destroy(ctx);
}
```

---

## Supported Providers

| Provider | Factory | Env Var |
|----------|---------|---------|
| Anthropic | `create_anthropic` | `ANTHROPIC_API_KEY` (`ANTHROPIC_AUTH_TOKEN` for Bearer) |
| OpenAI | `create_openai` | `OPENAI_API_KEY` |
| Google | `create_google` | `GOOGLE_API_KEY` |
| Amazon Bedrock | `create_bedrock` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
| DeepSeek | `create_deepseek` | `DEEPSEEK_API_KEY` |
| MoonshotAI | `create_moonshotai` | `MOONSHOT_API_KEY` |

All providers share the same interface. Switch models with a one-line change. For the full provider × protocol compatibility matrix (including z.ai/GLM and DeepSeek's dual endpoints), see [providers.md](providers.md).

---

## What Can You Build?

| Project | Complexity | Example |
|---------|-----------|---------|
| **CLI coding agent** (Claude Code clone) | ~50-150 lines | Give it file/shell tools, point at a codebase |
| **Code review bot** | ~40 lines | Read PR diffs, generate review comments |
| **Research agent** | ~60 lines | Web search + summarize tools |
| **DevOps agent** | ~80 lines | kubectl/terraform/aws tools |
| **Webapp backend** | ~100 lines | REST API that dispatches to agent |

---

## Building from Source

### Prerequisites

| Requirement | Minimum |
|-------------|---------|
| CMake | 3.20 |
| Boost | 1.82 (Beast, JSON, Asio) |
| OpenSSL | 1.1+ |
| C++ Compiler | GCC 12, Clang 15, or MSVC 2022 |

### Build

```bash
git clone https://github.com/anthropics/ai-sdk-cpp.git
cd ai-sdk-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
ctest   # run tests
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `AI_SDK_BUILD_SHARED` | ON | Build shared library |
| `AI_SDK_BUILD_STATIC` | ON | Build static library |
| `AI_SDK_BUILD_CLI` | ON | Build CLI agent example |
| `AI_SDK_BUILD_BINDINGS` | ON | Build C shared library |
| `AI_SDK_BUILD_PYTHON` | OFF | Build Python pybind11 module |
| `AI_SDK_BUILD_TESTS` | ON | Build test suite |
| `AI_SDK_PROVIDER_*` | ON | Toggle individual providers |

---

## Using the SDK

### From C++ (CMake)

```cmake
find_package(ai-sdk-cpp REQUIRED)
target_link_libraries(my_agent PRIVATE ai-sdk::core ai-sdk::anthropic)
```

### From Python

```bash
pip install ai-sdk-cpp
```

### From TypeScript / Node.js

```bash
npm install ai-sdk-cpp
```

### From Rust / Go / Any Language

Link against `libai_sdk` (the C shared library) and include `ai_sdk.h`. See the language-specific guides below.

---

## Guides

Step-by-step tutorials for building a coding agent from scratch:

| Language | Guide | Lines of code |
|----------|-------|---------------|
| C++ | [Building a Coding Agent (C++)](guides/building-a-coding-agent-cpp.md) | ~150 |
| Python | [Building a Coding Agent (Python)](guides/building-a-coding-agent-python.md) | ~80 |
| TypeScript | [Building a Coding Agent (TypeScript)](guides/building-a-coding-agent-typescript.md) | ~70 |
| Rust | [Building a Coding Agent (Rust)](guides/building-a-coding-agent-rust.md) | ~100 |
| Go | [Building a Coding Agent (Go)](guides/building-a-coding-agent-go.md) | ~90 |

---

## License

MIT — see [LICENSE](../LICENSE) for details.
