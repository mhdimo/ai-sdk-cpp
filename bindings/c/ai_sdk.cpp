#include "ai_sdk.h"
#include <ai/ai.hpp>
#include <ai/memory/memory.hpp>
#include <ai/session/context_strategy.hpp>
#include <ai/util/token_count.hpp>
#include <ai/core/generate_text.hpp>
#if defined(AI_SDK_PROVIDER_ANTHROPIC)
#include <ai/providers/anthropic/anthropic.hpp>
#endif
#if defined(AI_SDK_PROVIDER_OPENAI)
#include <ai/providers/openai/openai.hpp>
#endif
#if defined(AI_SDK_PROVIDER_GOOGLE)
#include <ai/providers/google/google.hpp>
#endif
#if defined(AI_SDK_PROVIDER_DEEPSEEK)
#include <ai/providers/deepseek/deepseek.hpp>
#endif
#if defined(AI_SDK_PROVIDER_ZAI)
#include <ai/providers/zai/zai.hpp>
#endif
#include <boost/asio.hpp>
#include <boost/json.hpp>
#include <chrono>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstring>
#include <cstdlib>

namespace json = boost::json;

struct ai_context {
    boost::asio::io_context ioc;
    std::string last_error;
};

struct ai_provider {
    ai::ProviderPtr ptr;
    ai_context* ctx;
};

struct ai_model {
    ai::LanguageModelPtr ptr;
    ai_context* ctx;
};

struct ai_tool_set {
    ai::ToolSet tools;
    struct ToolBinding {
        ai_tool_fn callback;
        void* user_data;
    };
    std::unordered_map<std::string, ToolBinding> bindings;
};

struct ai_agent {
    ai::ToolLoopAgent agent;
    ai_context* ctx;
};

struct ai_batch {
    ai::batch::BatchProcessorPtr ptr;
    ai_context* ctx;
};

// Backing storage for ai_batch_result_t: owns all strings the result points
// into, plus the items array. Freed by ai_batch_result_free.
struct BatchResultStorage {
    std::string batch_id;
    std::string status;
    std::vector<std::string> custom_ids;
    std::vector<std::string> result_jsons;
    std::vector<std::string> errors;
    std::vector<ai_batch_item_t> items;
};

static thread_local std::string g_result_text;
static thread_local std::string g_result_reason;

static ai_status_t map_exception(ai_context* ctx, const std::exception& e) {
    ctx->last_error = e.what();
    if (dynamic_cast<const ai::error::RateLimitError*>(&e)) return AI_ERROR_RATE_LIMIT;
    if (dynamic_cast<const ai::error::AuthenticationError*>(&e)) return AI_ERROR_AUTHENTICATION;
    if (dynamic_cast<const ai::error::TimeoutError*>(&e)) return AI_ERROR_TIMEOUT;
    if (dynamic_cast<const ai::error::StreamError*>(&e)) return AI_ERROR_STREAM;
    if (dynamic_cast<const ai::error::InvalidResponseError*>(&e)) return AI_ERROR_INVALID_RESPONSE;
    if (dynamic_cast<const ai::error::ApiCallError*>(&e)) return AI_ERROR_API_CALL;
    return AI_ERROR_INTERNAL;
}

// Drives an AsyncGenerator<StreamPart> to completion on `ioc`, dispatching each
// part to the C callback. Returns AI_OK, or a mapped error status after emitting
// a terminal AI_STREAM_ERROR event. Shared by ai_stream_text and
// ai_agent_call_stream (generator.next() must be awaited from a coroutine).
static ai_status_t consume_stream_to_callback(
    ai::AsyncGenerator<ai::StreamPart> stream,
    boost::asio::io_context& ioc,
    ai_stream_callback_fn cb,
    void* ud,
    ai_context* ctx
) {
    auto consume = [](ai::AsyncGenerator<ai::StreamPart> s,
                      ai_stream_callback_fn callback,
                      void* user_data) -> ai::Task<void> {
        while (auto part = co_await s.next()) {
            ai_stream_event_t event{};
            std::visit([&](auto&& p) {
                using T = std::decay_t<decltype(p)>;
                if constexpr (std::is_same_v<T, ai::TextDelta>) {
                    event.type = AI_STREAM_TEXT_DELTA;
                    event.text = p.delta.c_str();
                } else if constexpr (std::is_same_v<T, ai::ToolInputStart>) {
                    event.type = AI_STREAM_TOOL_CALL_START;
                    event.tool_name = p.tool_name.c_str();
                    event.tool_call_id = p.id.c_str();
                } else if constexpr (std::is_same_v<T, ai::ToolInputDelta>) {
                    event.type = AI_STREAM_TOOL_CALL_DELTA;
                    event.text = p.delta.c_str();
                    event.tool_call_id = p.id.c_str();
                } else if constexpr (std::is_same_v<T, ai::ToolInputEnd>) {
                    event.type = AI_STREAM_TOOL_CALL_END;
                    event.tool_call_id = p.id.c_str();
                } else if constexpr (std::is_same_v<T, ai::FinishPart>) {
                    event.type = AI_STREAM_FINISH;
                    switch (p.reason) {
                    case ai::FinishReason::Stop: event.finish_reason = "stop"; break;
                    case ai::FinishReason::Length: event.finish_reason = "length"; break;
                    case ai::FinishReason::ToolCalls: event.finish_reason = "tool_calls"; break;
                    default: event.finish_reason = "other"; break;
                    }
                    event.input_tokens = p.usage.input_tokens.total.value_or(0);
                    event.output_tokens = p.usage.output_tokens.total.value_or(0);
                } else if constexpr (std::is_same_v<T, ai::ReasoningStart>) {
                    event.type = AI_STREAM_REASONING_START;
                } else if constexpr (std::is_same_v<T, ai::ReasoningDelta>) {
                    event.type = AI_STREAM_REASONING_DELTA;
                    event.text = p.delta.c_str();
                } else if constexpr (std::is_same_v<T, ai::ReasoningEnd>) {
                    event.type = AI_STREAM_REASONING_END;
                } else if constexpr (std::is_same_v<T, ai::ErrorPart>) {
                    // Surface as a hard error so the call returns a non-OK status;
                    // the catch below emits the single AI_STREAM_ERROR event.
                    throw ai::error::StreamError(p.message);
                }
            }, *part);
            callback(event, user_data);
        }
    }(std::move(stream), cb, ud);

    try {
        consume.start();
        while (!consume.done()) {
            ioc.run_one();
        }
        consume.get();
        return AI_OK;
    } catch (const std::exception& e) {
        ai_stream_event_t err_event{};
        err_event.type = AI_STREAM_ERROR;
        err_event.text = e.what();
        cb(err_event, ud);
        return map_exception(ctx, e);
    }
}

/* -------------------------------------------------------------------------- */

extern "C" {

const char* ai_status_message(ai_status_t status) {
    switch (status) {
    case AI_OK: return "OK";
    case AI_ERROR_INVALID_ARGUMENT: return "Invalid argument";
    case AI_ERROR_API_CALL: return "API call failed";
    case AI_ERROR_RATE_LIMIT: return "Rate limited";
    case AI_ERROR_AUTHENTICATION: return "Authentication failed";
    case AI_ERROR_TIMEOUT: return "Request timed out";
    case AI_ERROR_STREAM: return "Stream error";
    case AI_ERROR_CANCELLED: return "Cancelled";
    case AI_ERROR_INVALID_RESPONSE: return "Invalid response";
    case AI_ERROR_INTERNAL: return "Internal error";
    }
    return "Unknown error";
}

const char* ai_last_error(ai_context_t ctx) {
    if (!ctx) return "";
    return ctx->last_error.c_str();
}

ai_context_t ai_context_create(void) {
    return new ai_context{};
}

void ai_context_destroy(ai_context_t ctx) {
    delete ctx;
}

ai_provider_t ai_provider_create(ai_context_t ctx, const char* provider_name, ai_provider_options_t opts) {
    if (!ctx || !provider_name) return nullptr;

    std::string name(provider_name);
    std::optional<std::string> api_key;
    if (opts.api_key) api_key = std::string(opts.api_key);

    ai::ProviderPtr provider;

    try {
#if defined(AI_SDK_PROVIDER_ANTHROPIC)
        if (!provider && name == "anthropic") {
            ai::providers::anthropic::AnthropicOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            else if (const char* env = std::getenv("ANTHROPIC_BASE_URL"); env && *env) o.base_url = env;
            provider = ai::providers::anthropic::create_anthropic(std::move(o));
        }
#endif
#if defined(AI_SDK_PROVIDER_OPENAI)
        if (!provider && name == "openai") {
            ai::providers::openai::OpenAIOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            else if (const char* env = std::getenv("OPENAI_BASE_URL"); env && *env) o.base_url = env;
            provider = ai::providers::openai::create_openai(std::move(o));
        }
#endif
#if defined(AI_SDK_PROVIDER_GOOGLE)
        if (!provider && name == "google") {
            ai::providers::google::GoogleOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            provider = ai::providers::google::create_google(std::move(o));
        }
#endif
#if defined(AI_SDK_PROVIDER_DEEPSEEK)
        if (!provider && name == "deepseek") {
            ai::providers::deepseek::DeepSeekOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::deepseek::create_deepseek(std::move(o));
        }
        if (!provider && name == "deepseek-anthropic") {
            ai::providers::deepseek::DeepSeekAnthropicOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::deepseek::create_deepseek_anthropic(std::move(o));
        }
#endif
#if defined(AI_SDK_PROVIDER_ZAI)
        if (!provider && name == "zai") {
            ai::providers::zai::ZaiOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::zai::create_zai(std::move(o));
        }
        if (!provider && name == "zai-openai") {
            ai::providers::zai::ZaiOpenAiOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::zai::create_zai_openai(std::move(o));
        }
#endif
        if (!provider) {
            ctx->last_error = "Unknown or not-built provider: " + name;
            return nullptr;
        }
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return nullptr;
    }

    auto p = new ai_provider{};
    p->ptr = std::move(provider);
    p->ctx = ctx;
    return p;
}

void ai_provider_destroy(ai_provider_t provider) {
    delete provider;
}

ai_model_t ai_model_create(ai_provider_t provider, const char* model_id) {
    if (!provider || !model_id) return nullptr;
    auto m = new ai_model{};
    m->ptr = provider->ptr->language_model(model_id);
    m->ctx = provider->ctx;
    return m;
}

void ai_model_destroy(ai_model_t model) {
    delete model;
}

ai_tool_set_t ai_tool_set_create(void) {
    return new ai_tool_set{};
}

void ai_tool_set_destroy(ai_tool_set_t tools) {
    delete tools;
}

ai_status_t ai_tool_set_add(
    ai_tool_set_t tools,
    const char* name,
    const char* description,
    const char* input_schema_json,
    ai_tool_fn callback,
    void* user_data
) {
    if (!tools || !name || !input_schema_json || !callback) {
        return AI_ERROR_INVALID_ARGUMENT;
    }

    std::string tool_name(name);
    tools->bindings[tool_name] = {callback, user_data};

    auto schema_val = json::parse(input_schema_json);
    ai::schema::JsonSchema schema(schema_val.as_object());

    auto* binding = &tools->bindings[tool_name];

    tools->tools.add(ai::tool(
        tool_name,
        schema,
        description ? std::string(description) : "",
        [binding, tool_name](json::value input, ai::ToolExecutionContext) -> ai::Task<json::value> {
            std::string input_str = json::serialize(input);
            auto result = binding->callback(tool_name.c_str(), input_str.c_str(), binding->user_data);

            json::value output;
            if (result.output_json) {
                boost::system::error_code ec;
                output = json::parse(result.output_json, ec);
                if (ec) output = json::value(result.output_json);
            }

            if (result.is_error) {
                throw std::runtime_error(result.output_json ? result.output_json : "Tool error");
            }

            co_return output;
        }
    ));

    return AI_OK;
}

ai_status_t ai_generate_text(ai_generate_options_t opts, ai_generate_result_t* result) {
    if (!opts.model || !result) return AI_ERROR_INVALID_ARGUMENT;
    auto* ctx = opts.model->ctx;

    try {
        ai::Prompt prompt;
        if (opts.system) {
            prompt.push_back(ai::SystemMessage{.content = std::string(opts.system)});
        }

        if (opts.messages_json) {
            auto messages = json::parse(opts.messages_json);
            auto deserialized = ai::workflow::deserialize_prompt(messages.as_array());
            for (auto& m : deserialized) prompt.push_back(std::move(m));
        } else if (opts.prompt) {
            ai::UserContent content;
            content.push_back(ai::TextPart{.text = std::string(opts.prompt)});
            prompt.push_back(ai::UserMessage{.content = std::move(content)});
        }

        ai::GenerateTextOptions gen_opts;
        gen_opts.model = opts.model->ptr;
        gen_opts.messages = std::move(prompt);
        gen_opts.max_steps = opts.max_steps > 0 ? opts.max_steps : 1;
        if (opts.max_output_tokens > 0) gen_opts.max_output_tokens = opts.max_output_tokens;
        if (opts.temperature >= 0) gen_opts.temperature = opts.temperature;
        if (opts.tools) gen_opts.tools = opts.tools->tools;

        auto task = ai::generate_text(std::move(gen_opts));
        task.start();
        while (!task.done()) {
            ctx->ioc.run_one();
        }
        auto gen_result = task.get();

        g_result_text = gen_result.text;
        switch (gen_result.finish_reason) {
        case ai::FinishReason::Stop: g_result_reason = "stop"; break;
        case ai::FinishReason::Length: g_result_reason = "length"; break;
        case ai::FinishReason::ToolCalls: g_result_reason = "tool_calls"; break;
        default: g_result_reason = "other"; break;
        }

        result->text = g_result_text.c_str();
        result->finish_reason = g_result_reason.c_str();
        result->input_tokens = gen_result.usage.input_tokens.total.value_or(0);
        result->output_tokens = gen_result.usage.output_tokens.total.value_or(0);
        result->steps = static_cast<int>(gen_result.steps.size());

        return AI_OK;
    } catch (const std::exception& e) {
        return map_exception(ctx, e);
    }
}

void ai_generate_result_free(ai_generate_result_t* result) {
    if (result) {
        result->text = nullptr;
        result->finish_reason = nullptr;
    }
}

ai_status_t ai_stream_text(ai_generate_options_t opts, ai_stream_callback_fn callback, void* user_data) {
    if (!opts.model || !callback) return AI_ERROR_INVALID_ARGUMENT;
    auto* ctx = opts.model->ctx;

    try {
        ai::Prompt prompt;
        if (opts.system) {
            prompt.push_back(ai::SystemMessage{.content = std::string(opts.system)});
        }
        if (opts.messages_json) {
            auto messages = json::parse(opts.messages_json);
            auto deserialized = ai::workflow::deserialize_prompt(messages.as_array());
            for (auto& m : deserialized) prompt.push_back(std::move(m));
        } else if (opts.prompt) {
            ai::UserContent content;
            content.push_back(ai::TextPart{.text = std::string(opts.prompt)});
            prompt.push_back(ai::UserMessage{.content = std::move(content)});
        }

        ai::CallOptions call_opts;
        call_opts.prompt = std::move(prompt);
        if (opts.max_output_tokens > 0) call_opts.max_output_tokens = opts.max_output_tokens;
        if (opts.temperature >= 0) call_opts.temperature = opts.temperature;

        auto task = opts.model->ptr->do_stream(std::move(call_opts));
        task.start();
        while (!task.done()) {
            ctx->ioc.run_one();
        }
        auto stream_result = task.get();
        return consume_stream_to_callback(
            std::move(stream_result.stream), ctx->ioc, callback, user_data, ctx);
    } catch (const std::exception& e) {
        ai_stream_event_t err_event{};
        err_event.type = AI_STREAM_ERROR;
        err_event.text = e.what();
        callback(err_event, user_data);
        return map_exception(ctx, e);
    }
}

ai_agent_t ai_agent_create(ai_agent_options_t opts) {
    if (!opts.model || !opts.tools) return nullptr;

    ai::ToolLoopAgentOptions agent_opts{
        .model = opts.model->ptr,
        .tools = std::move(opts.tools->tools),
        .instructions = opts.instructions ? std::string(opts.instructions) : "",
        .max_steps = opts.max_steps > 0 ? opts.max_steps : 50,
    };

    if (opts.on_event) {
        auto cb = opts.on_event;
        auto ud = opts.user_data;
        agent_opts.on_step_finish = [cb, ud](const ai::StepResult& step) {
            auto text = step.result.text();
            if (!text.empty()) {
                ai_stream_event_t event{};
                event.type = AI_STREAM_TEXT_DELTA;
                event.text = text.c_str();
                cb(event, ud);
            }
            ai_stream_event_t step_event{};
            step_event.type = AI_STREAM_STEP_FINISH;
            cb(step_event, ud);
        };
    }

    auto a = new ai_agent{
        .agent = ai::ToolLoopAgent(std::move(agent_opts)),
        .ctx = opts.model->ctx,
    };
    return a;
}

void ai_agent_destroy(ai_agent_t agent) {
    delete agent;
}

ai_status_t ai_agent_call(ai_agent_t agent, const char* prompt, ai_generate_result_t* result) {
    if (!agent || !prompt || !result) return AI_ERROR_INVALID_ARGUMENT;

    try {
        auto task = agent->agent.call(std::string(prompt));
        task.start();
        while (!task.done()) {
            agent->ctx->ioc.run_one();
        }
        auto agent_result = task.get();

        g_result_text = agent_result.text;
        g_result_reason = "stop";

        result->text = g_result_text.c_str();
        result->finish_reason = g_result_reason.c_str();
        result->input_tokens = agent_result.usage.input_tokens.total.value_or(0);
        result->output_tokens = agent_result.usage.output_tokens.total.value_or(0);
        result->steps = static_cast<int>(agent_result.steps.size());

        return AI_OK;
    } catch (const std::exception& e) {
        return map_exception(agent->ctx, e);
    }
}

ai_status_t ai_agent_call_stream(ai_agent_t agent, const char* prompt, ai_stream_callback_fn callback, void* user_data) {
    if (!agent || !prompt || !callback) return AI_ERROR_INVALID_ARGUMENT;
    auto* ctx = agent->ctx;

    try {
        // Run the agent's streaming tool loop, then dispatch each stream part
        // (text/tool/finish) to the C callback as it is produced.
        auto outer = agent->agent.stream(std::string(prompt));
        outer.start();
        while (!outer.done()) {
            ctx->ioc.run_one();
        }
        auto result = outer.get();

        return consume_stream_to_callback(
            std::move(result.stream), ctx->ioc, callback, user_data, ctx);
    } catch (const std::exception& e) {
        ai_stream_event_t err_event{};
        err_event.type = AI_STREAM_ERROR;
        err_event.text = e.what();
        callback(err_event, user_data);
        return map_exception(ctx, e);
    }
}

ai_batch_t ai_batch_create(ai_provider_t provider, const char* model_id) {
    if (!provider || !model_id) return nullptr;
    auto proc = provider->ptr->batch_processor(model_id);
    if (!proc) {
        provider->ctx->last_error =
            "Provider does not support batching: " + std::string(provider->ptr->provider_id());
        return nullptr;
    }
    return new ai_batch{.ptr = std::move(proc), .ctx = provider->ctx};
}

void ai_batch_destroy(ai_batch_t batch) {
    delete batch;
}

ai_status_t ai_batch_run(
    ai_batch_t batch,
    const ai_batch_request_t* requests,
    int count,
    int poll_interval_ms,
    ai_batch_result_t* result
) {
    if (!batch || !result) return AI_ERROR_INVALID_ARGUMENT;
    if (count < 0) return AI_ERROR_INVALID_ARGUMENT;
    if (count > 0 && !requests) return AI_ERROR_INVALID_ARGUMENT;
    auto* ctx = batch->ctx;

    try {
        std::vector<ai::batch::BatchRequest> reqs;
        reqs.reserve(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            const auto& r = requests[i];
            ai::Prompt prompt;
            if (r.system) {
                prompt.push_back(ai::SystemMessage{.content = std::string(r.system)});
            }
            if (r.prompt) {
                ai::UserContent uc;
                uc.push_back(ai::TextPart{.text = std::string(r.prompt)});
                prompt.push_back(ai::UserMessage{.content = std::move(uc)});
            }
            ai::CallOptions co;
            co.prompt = std::move(prompt);
            if (r.max_output_tokens > 0) co.max_output_tokens = r.max_output_tokens;
            if (r.temperature >= 0) co.temperature = r.temperature;
            reqs.push_back(ai::batch::BatchRequest{
                .custom_id = std::string(r.custom_id ? r.custom_id : ""),
                .options = std::move(co),
            });
        }

        ai::batch::BatchRunOptions opts{
            .processor = batch->ptr,
            .requests = std::move(reqs),
            .io_context = ctx->ioc,
            .poll_interval = std::chrono::milliseconds(poll_interval_ms > 0 ? poll_interval_ms : 5000),
        };

        auto task = ai::batch::run_batch(std::move(opts));
        task.start();
        while (!task.done()) {
            ctx->ioc.run_one();
        }
        auto run_result = task.get();

        auto* storage = new BatchResultStorage();
        storage->batch_id = std::move(run_result.batch_id);
        switch (run_result.final_status.status) {
        case ai::batch::BatchStatus::Completed: storage->status = "completed"; break;
        case ai::batch::BatchStatus::Failed: storage->status = "failed"; break;
        case ai::batch::BatchStatus::Cancelled: storage->status = "cancelled"; break;
        case ai::batch::BatchStatus::Expired: storage->status = "expired"; break;
        default: storage->status = "other"; break;
        }

        auto n = run_result.results.size();
        storage->custom_ids.reserve(n);
        storage->result_jsons.reserve(n);
        storage->errors.reserve(n);
        storage->items.reserve(n);
        for (auto& item : run_result.results) {
            storage->custom_ids.push_back(item.custom_id);
            if (item.error) {
                storage->result_jsons.emplace_back();
                storage->errors.push_back(*item.error);
            } else {
                std::string text = item.result ? item.result->text() : std::string{};
                storage->result_jsons.push_back(std::move(text));
                storage->errors.emplace_back();
            }
            // Pointers are stable: the three string vectors are reserved above.
            storage->items.push_back(ai_batch_item_t{
                .custom_id = storage->custom_ids.back().c_str(),
                .result_json = storage->result_jsons.back().empty()
                    ? nullptr : storage->result_jsons.back().c_str(),
                .error = storage->errors.back().empty()
                    ? nullptr : storage->errors.back().c_str(),
            });
        }

        result->batch_id = storage->batch_id.c_str();
        result->items = storage->items.data();
        result->count = static_cast<int>(storage->items.size());
        result->status = storage->status.c_str();
        result->_storage = storage;

        return AI_OK;
    } catch (const std::exception& e) {
        return map_exception(ctx, e);
    }
}

void ai_batch_result_free(ai_batch_result_t* result) {
    if (result && result->_storage) {
        delete static_cast<BatchResultStorage*>(result->_storage);
        result->_storage = nullptr;
        result->items = nullptr;
        result->batch_id = nullptr;
        result->status = nullptr;
        result->count = 0;
    }
}

// --- Session + standard toolkit + permissions -----------------------------

struct ai_session {
    ai::Session session;
    ai_context* ctx;
};

ai_session_t ai_session_create(ai_agent_t agent) {
    if (!agent) return nullptr;
    return new ai_session{.session = ai::Session(agent->agent), .ctx = agent->ctx};
}

ai_session_t ai_session_create_with_memory(ai_agent_t agent, const char* memory_dir, int max_context_tokens) {
    if (!agent || !memory_dir) return nullptr;
    // MemoryContextStrategy over a sliding window: inject relevant persisted
    // memory before each turn; the inner SlidingWindowStrategy auto-compacts
    // (pair-safe) near max_context_tokens.
    auto store = std::make_shared<ai::memory::MarkdownMemoryStore>(memory_dir);
    auto retriever = std::make_shared<ai::memory::KeywordRetriever>(*store);
    auto inner = std::make_shared<ai::SlidingWindowStrategy>();
    auto strategy = std::make_shared<ai::memory::MemoryContextStrategy>(
        store, retriever, inner);
    ai::ContextWindow window{};
    window.max_tokens = max_context_tokens > 0 ? max_context_tokens : (128 * 1024);
    window.reserved_output_tokens = 4096;
    auto counter = std::make_shared<ai::ApproximateTokenCounter>();
    ai::Session session(agent->agent, window, counter, strategy);

    // Auto-checkpoint: every 5 turns, summarize the conversation into a memory
    // record (the "continuously improves itself" loop). Best-effort.
    if (agent->agent.model()) {
        auto summarizer = [model = agent->agent.model()](const ai::Prompt& history) -> ai::Task<std::string> {
            ai::GenerateTextOptions o;
            o.model = model;
            o.system = "Summarize the conversation so far into a concise checkpoint: "
                       "key decisions, current state, and next steps.";
            std::string transcript;
            for (auto& m : history) {
                std::visit([&](auto&& msg) {
                    using T = std::decay_t<decltype(msg)>;
                    if constexpr (std::is_same_v<T, ai::UserMessage>) {
                        for (auto& p : msg.content)
                            if (auto* t = std::get_if<ai::TextPart>(&p)) transcript += "User: " + t->text + "\n";
                    } else if constexpr (std::is_same_v<T, ai::AssistantMessage>) {
                        for (auto& p : msg.content)
                            if (auto* t = std::get_if<ai::TextPart>(&p)) transcript += "Assistant: " + t->text + "\n";
                    }
                }, m);
            }
            o.prompt = transcript;
            auto r = co_await ai::generate_text(std::move(o));
            co_return r.text;
        };
        session.set_on_turn_finish(
            ai::memory::make_checkpoint_writer(store, summarizer, 5));
    }

    return new ai_session{ .session = std::move(session), .ctx = agent->ctx };
}

void ai_session_destroy(ai_session_t session) {
    delete session;
}

ai_status_t ai_session_send(ai_session_t session, const char* prompt, ai_generate_result_t* result) {
    if (!session || !prompt || !result) return AI_ERROR_INVALID_ARGUMENT;
    auto* ctx = session->ctx;
    try {
        auto task = session->session.send(std::string(prompt));
        task.start();
        while (!task.done()) {
            ctx->ioc.run_one();
        }
        auto r = task.get();
        g_result_text = r.text;
        switch (r.finish_reason) {
        case ai::FinishReason::Stop: g_result_reason = "stop"; break;
        case ai::FinishReason::Length: g_result_reason = "length"; break;
        case ai::FinishReason::ToolCalls: g_result_reason = "tool_calls"; break;
        default: g_result_reason = "other"; break;
        }
        result->text = g_result_text.c_str();
        result->finish_reason = g_result_reason.c_str();
        result->input_tokens = r.usage.input_tokens.total.value_or(0);
        result->output_tokens = r.usage.output_tokens.total.value_or(0);
        result->steps = static_cast<int>(r.steps.size());
        return AI_OK;
    } catch (const std::exception& e) {
        return map_exception(ctx, e);
    }
}

static void append_turn_messages(ai::Session& session, const ai::GenerateTextResult& result) {
    ai::Prompt delta;

    for (const auto& step : result.steps) {
        const auto& res = step.result;
        const auto tool_calls_content = res.tool_calls();
        if (!tool_calls_content.empty()) {
            ai::AssistantContent content;
            for (auto& c : res.content) {
                if (auto* text = std::get_if<ai::TextContent>(&c)) {
                    if (!text->text.empty()) {
                        content.push_back(ai::TextPart{.text = text->text});
                    }
                } else if (auto* reasoning = std::get_if<ai::ReasoningContent>(&c)) {
                    ai::ReasoningPart rp{.text = reasoning->text};
                    rp.signature = reasoning->signature;
                    rp.redacted_data = reasoning->redacted_data;
                    content.push_back(std::move(rp));
                } else if (auto* tc = std::get_if<ai::ToolCallContent>(&c)) {
                    content.push_back(ai::ToolCallPart{
                        .tool_call_id = tc->tool_call_id,
                        .tool_name = tc->tool_name,
                        .input = ai::json::safe_parse(tc->input)
                                     .value_or(boost::json::value(tc->input)),
                    });
                }
            }
            if (!content.empty()) {
                delta.push_back(ai::AssistantMessage{.content = std::move(content)});
            }

            ai::ToolContent tc_content;
            for (auto& r : step.tool_results) {
                ai::ToolResultOutput output = r.is_error
                    ? ai::ToolResultOutput{ai::ErrorJsonOutput{.value = r.output}}
                    : ai::ToolResultOutput{ai::JsonOutput{.value = r.output}};
                tc_content.push_back(ai::ToolResultPart{
                    .tool_call_id = r.tool_call_id,
                    .tool_name = r.tool_name,
                    .output = std::move(output),
                });
            }
            if (!tc_content.empty()) {
                delta.push_back(ai::ToolMessage{.content = std::move(tc_content)});
            }
        }
    }

    bool ends_with_assistant_text = false;
    if (!delta.empty()) {
        if (auto* asst = std::get_if<ai::AssistantMessage>(&delta.back())) {
            for (auto& part : asst->content) {
                if (auto* t = std::get_if<ai::TextPart>(&part)) {
                    if (!t->text.empty()) {
                        ends_with_assistant_text = true;
                    }
                }
            }
        }
    }
    if (!ends_with_assistant_text && !result.text.empty()) {
        ai::AssistantContent content;
        content.push_back(ai::TextPart{.text = result.text});
        delta.push_back(ai::AssistantMessage{.content = std::move(content)});
    }

    auto snapshot = session.snapshot();
    for (auto& m : delta) {
        snapshot.history.push_back(std::move(m));
    }
    session.restore(std::move(snapshot));
}

ai_status_t ai_session_send_stream(
    ai_session_t session,
    const char* prompt,
    ai_stream_callback_fn callback,
    void* user_data
) {
    if (!session || !prompt || !callback) return AI_ERROR_INVALID_ARGUMENT;
    auto* ctx = session->ctx;

    try {
        auto outer = session->session.send_stream(std::string(prompt));
        outer.start();
        while (!outer.done()) {
            ctx->ioc.run_one();
        }
        auto result = outer.get();

        auto status = consume_stream_to_callback(
            std::move(result.stream), ctx->ioc, callback, user_data, ctx);
        if (status != AI_OK) return status;

        auto full_result_task = std::move(result.full_result);
        full_result_task.start();
        while (!full_result_task.done()) {
            ctx->ioc.run_one();
        }
        auto gen_result = full_result_task.get();

        // Dispatch executed tool results to callback
        for (const auto& step : gen_result.steps) {
            for (const auto& tr : step.tool_results) {
                std::string output_str;
                if (tr.output.is_string()) {
                    output_str = tr.output.as_string();
                } else {
                    output_str = boost::json::serialize(tr.output);
                }
                ai_stream_event_t event{};
                event.type = AI_STREAM_TOOL_RESULT;
                event.tool_name = tr.tool_name.c_str();
                event.tool_call_id = tr.tool_call_id.c_str();
                event.text = output_str.c_str();
                callback(event, user_data);
            }
        }

        append_turn_messages(session->session, gen_result);

        // Fire the post-turn hook (e.g. checkpoint writer) — best-effort.
        {
            auto hook = session->session.fire_turn_finish();
            hook.start();
            while (!hook.done()) ctx->ioc.run_one();
            try { hook.get(); } catch (...) {}
        }

        return AI_OK;
    } catch (const std::exception& e) {
        return map_exception(ctx, e);
    }
}


ai_tool_set_t ai_standard_toolkit_create(void) {
    auto* ts = new ai_tool_set{};
    ts->tools = ai::standard_toolkit();
    return ts;
}

ai_tool_set_t ai_with_permissions(ai_tool_set_t tools, ai_permission_policy_fn policy, void* user_data) {
    if (!tools || !policy) return nullptr;
    auto* out = new ai_tool_set{};
    ai::PermissionPolicy p = [policy, user_data](const std::string& tool,
                                                  const boost::json::value& input) {
        std::string input_str = boost::json::serialize(input);
        int d = policy(tool.c_str(), input_str.c_str(), user_data);
        switch (d) {
        case AI_PERMISSION_ALLOW: return ai::PermissionDecision::Allow;
        case AI_PERMISSION_DENY: return ai::PermissionDecision::Deny;
        default: return ai::PermissionDecision::Ask;
        }
    };
    out->tools = ai::with_permissions(std::move(tools->tools), p);
    return out;
}

// --- Memory store ---

struct ai_memory_store {
    ai::memory::MarkdownMemoryStore store;
};

ai_memory_store_t ai_memory_store_create(const char* dir) {
    return new ai_memory_store{
        .store = ai::memory::MarkdownMemoryStore(dir ? dir : ".agent/memory")};
}

void ai_memory_store_destroy(ai_memory_store_t store) {
    delete store;
}

ai_status_t ai_memory_save(ai_memory_store_t store, const char* scope,
                           const char* key, const char* content) {
    if (!store || !scope || !content) return AI_ERROR_INVALID_ARGUMENT;
    try {
        ai::memory::MemoryRecord rec;
        rec.scope = scope;
        rec.key = key ? key : "";
        rec.content = content;
        store->store.add(std::move(rec));
        return AI_OK;
    } catch (const std::exception& e) {
        return AI_ERROR_INTERNAL;
    }
}

const char* ai_sdk_version(void) {
    return "0.1.0";
}

} // extern "C"
