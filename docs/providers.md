# Providers & Compatibility Matrix

The SDK talks to LLM providers through two base protocol implementations:

- **OpenAI Chat Completions** — `providers/openai/`. Serves OpenAI (native), DeepSeek (OpenAI endpoint), MoonshotAI, and z.ai's OpenAI endpoint.
- **Anthropic Messages** — `providers/anthropic/`. Serves Anthropic (native), z.ai/GLM (Anthropic endpoint), and DeepSeek (Anthropic endpoint).

DeepSeek and MoonshotAI ship as thin wrappers that instantiate the OpenAI provider with a different `base_url` and a provider-specific API-key env var (see `providers/deepseek/src/deepseek_provider.cpp` and `providers/moonshotai/src/moonshotai_provider.cpp`). z.ai is reached by pointing either base provider at a z.ai base URL.

> **Removed providers.** The minor OpenAI-compatible wrappers — **Groq, xAI, Mistral, Fireworks, TogetherAI, Perplexity, Cohere** — have been removed. The kept set is: **OpenAI, Anthropic, DeepSeek, Amazon Bedrock, Google, MoonshotAI**. For any other OpenAI-compatible host, use the generic `openai_compatible` provider (`providers/openai-compatible/`) with a custom `base_url`.

## Kept providers

| Provider | Base protocol | Wraps | Env var |
|---|---|---|---|
| OpenAI | OpenAI | native | `OPENAI_API_KEY` |
| Anthropic | Anthropic | native | `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` |
| DeepSeek | OpenAI | `openai` + `base_url` | `DEEPSEEK_API_KEY` |
| Amazon Bedrock | native | native | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
| Google | native | native | `GOOGLE_API_KEY` |
| MoonshotAI | OpenAI | `openai` + `base_url` | `MOONSHOT_API_KEY` |

## Compatibility matrix

| Provider | Primary base | Endpoint | Auth | Thinking visible? |
|---|---|---|---|---|
| OpenAI | OpenAI | `api.openai.com/v1/chat/completions` | Bearer | no (token count only) |
| Anthropic | Anthropic | `api.anthropic.com/v1/messages` | x-api-key / Bearer | yes (`thinking` + signature) |
| DeepSeek / OpenAI | OpenAI | `api.deepseek.com/chat/completions` | Bearer | yes (`reasoning_content`) |
| DeepSeek / Anthropic | Anthropic | `api.deepseek.com/anthropic/v1/messages` | x-api-key | yes |
| z.ai / Anthropic | Anthropic | `api.z.ai/api/anthropic/v1/messages` | Bearer (+ x-api-key) | **no** (endpoint ignores thinking) |
| z.ai / OpenAI | OpenAI | `api.z.ai/api/paas/v4/chat/completions` | Bearer | yes (`reasoning_content`) |

---

## z.ai / GLM configuration

A common setup: drive GLM models through an Anthropic-compatible or OpenAI-compatible endpoint hosted by z.ai.

- **Anthropic-compatible base URL:** `https://api.z.ai/api/anthropic` (China: `https://open.bigmodel.cn/api/anthropic`).
- **Auth:** Bearer token. Set `auth_token` / `ANTHROPIC_AUTH_TOKEN` (**not** `api_key`) so Claude-Code-style clients authenticate. The Anthropic provider prefers `auth_token` and emits `Authorization: Bearer ...` when it is present, falling back to `x-api-key` only when `api_key` is set (see `providers/anthropic/src/anthropic_provider.cpp`).
- **Models:** `glm-5.2` (flagship, 1M context), `glm-4.7`, `glm-4.6`, `glm-4.5` / `glm-4.5-air`. Raw `claude-*` model names are auto-mapped.
- **The `[1m]` suffix is a client-side alias, not an API model id.** `glm-5.2[1m]` is a Claude-Code client convention; the z.ai API rejects it with `[1211] Unknown Model`. Pass a **bare** model id (e.g. `glm-5.2`). A `zai` provider strips trailing `[...]`; if you use the Anthropic provider directly, pass the bare id yourself.
- **Thinking blocks are not surfaced on the Anthropic endpoint.** `budget_tokens` is ignored. To observe reasoning, use the z.ai **OpenAI** endpoint (`https://api.z.ai/api/paas/v4`) and read `reasoning_content`.

### Example (Anthropic endpoint, Bearer auth)

```cpp
#include <ai/providers/anthropic/anthropic.hpp>

auto p = ai::providers::anthropic::create_anthropic({
    .auth_token = std::optional<std::string>{},      // or set ANTHROPIC_AUTH_TOKEN
    .base_url   = "https://api.z.ai/api/anthropic",
    .io_context = ioc,
});
auto model = p->language_model("glm-5.2");   // bare id — never "glm-5.2[1m]"
```

---

## DeepSeek

DeepSeek exposes both an OpenAI-compatible endpoint (default for the `deepseek` wrapper) and an Anthropic-compatible endpoint.

### OpenAI endpoint (default)

- **Base URL:** `https://api.deepseek.com` (wrapper default is `https://api.deepseek.com/v1`).
- **Models:** `deepseek-v4-flash` / `deepseek-v4-pro` (1M context). `deepseek-chat` / `deepseek-reasoner` are **deprecated 2026-07-24**.
- **Reasoning:** toggle with `provider_options.openai.thinking = "enabled" | "disabled"` (default **enabled**), and set the effort via `reasoning_effort` ∈ {`high`, `max`}. `reasoning_content` is returned *before* the answer. **Do not echo `reasoning_content` back** into subsequent multi-turn requests.
- **Structured output:** DeepSeek supports `json_object` only — **not** `json_schema`. The SDK auto-downgrades to `json_object` (plus prompt injection of the schema) on any non-OpenAI host, and falls back further to a prompt-based mode if `structured_output` is set to `prompt`. You can force this with `provider_options.openai.structured_output`.

### Anthropic endpoint

- **Base URL:** `https://api.deepseek.com/anthropic` (auth via `x-api-key`).
- `budget_tokens` and `cache_control` are **ignored** by this endpoint.

### Example (OpenAI endpoint)

```cpp
#include <ai/providers/deepseek/deepseek.hpp>

auto p = ai::providers::deepseek::create_deepseek({
    .base_url = "https://api.deepseek.com",
    .io_context = ioc,
});
auto model = p->language_model("deepseek-v4-pro");
```

---

## Provider controls via options

Cross-provider controls are conveyed through `ai::CallOptions` (`include/ai/model/call_options.hpp`):

- **`reasoning`** — level string, one of `minimal`, `low`, `medium`, `high`, `xhigh`, `none`. Mapped per provider:
  - **OpenAI** → `reasoning_effort`.
  - **Anthropic** → extended-thinking `budget_tokens`.
  - **DeepSeek** → `thinking` toggle + `reasoning_effort`.
- **`provider_options.openai`** — object with keys `{reasoning_effort, thinking, structured_output, json_schema_strict}` (see the OpenAI request builder in `providers/openai/src/openai_model.cpp`).
- **Extended-thinking signature round-trip (Anthropic):** Anthropic `thinking` blocks carry a `signature`. The SDK captures this signature and re-emits it in assistant turns, enabling correct multi-turn tool use with extended thinking.

For the full list of per-provider options and CMake toggles (`AI_SDK_PROVIDER_<NAME>`), see the root [README](../README.md).
