# ai-sdk-cpp

A high-performance C++20 AI agent framework.

Like **llama.cpp** is to inference, **ai-sdk-cpp** is to agent orchestration. One native engine, any language.

- **11 providers** — Anthropic, OpenAI, Google, Groq, xAI, Mistral, Fireworks, Together AI, Perplexity, Cohere, DeepSeek
- **Automatic tool loops** — define tools, the agent calls them until the task is done
- **Streaming** — SSE parsing, token-by-token output, backpressure
- **Any language** — C++, Python, TypeScript, Rust, Go, or anything with C FFI
- **C++20 coroutines** — native async I/O, no threads for simple use cases

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Your Agent (C++ / Python / TypeScript / Rust / Go / C) │
├─────────────────────────────────────────────────────────┤
│  Bindings  (pybind11 │ N-API │ cgo │ FFI │ C header)    │
├─────────────────────────────────────────────────────────┤
│  Core      (generate_text · stream_text · Agent loop)   │
├─────────┬────────┬────────┬────────┬────────┬───────────┤
│Anthropic│ OpenAI │ Google │  Groq  │  xAI   │ 6 more…   │
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

### Rust

```rust
use ai_sdk::{Context, Agent};

fn main() {
    let ctx = Context::new();
    let provider = ctx.provider("anthropic", None, None);
    let model = provider.model("claude-sonnet-4-20250514");

    let result = ai_sdk::generate_text(&model, "Hello!", None).unwrap();
    println!("{}", result.text);
}
```

### Go

```go
package main

import (
    "fmt"
    ai "your-module/aisdk"
)

func main() {
    ctx := ai.NewContext()
    defer ctx.Close()

    provider := ctx.NewProvider("anthropic", "", "")
    defer provider.Close()

    model := provider.NewModel("claude-sonnet-4-20250514")
    defer model.Close()

    result, _ := ai.GenerateText(model, "Hello!", nil)
    fmt.Println(result.Text)
}
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
| Anthropic | `create_anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI | `create_openai` | `OPENAI_API_KEY` |
| Google | `create_google` | `GOOGLE_API_KEY` |
| Groq | `create_groq` | `GROQ_API_KEY` |
| xAI | `create_xai` | `XAI_API_KEY` |
| Mistral | `create_mistral` | `MISTRAL_API_KEY` |
| Fireworks | `create_fireworks` | `FIREWORKS_API_KEY` |
| Together AI | `create_togetherai` | `TOGETHER_API_KEY` |
| Perplexity | `create_perplexity` | `PERPLEXITY_API_KEY` |
| Cohere | `create_cohere` | `COHERE_API_KEY` |
| DeepSeek | `create_deepseek` | `DEEPSEEK_API_KEY` |

All providers share the same interface. Switch models with a one-line change.

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

Apache-2.0
