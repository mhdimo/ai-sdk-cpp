#include "ai_sdk.h"
#include <ai/ai.hpp>
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
#if defined(AI_SDK_PROVIDER_GROQ)
#include <ai/providers/groq/groq.hpp>
#endif
#if defined(AI_SDK_PROVIDER_XAI)
#include <ai/providers/xai/xai.hpp>
#endif
#if defined(AI_SDK_PROVIDER_MISTRAL)
#include <ai/providers/mistral/mistral.hpp>
#endif
#if defined(AI_SDK_PROVIDER_FIREWORKS)
#include <ai/providers/fireworks/fireworks.hpp>
#endif
#if defined(AI_SDK_PROVIDER_TOGETHERAI)
#include <ai/providers/togetherai/togetherai.hpp>
#endif
#if defined(AI_SDK_PROVIDER_PERPLEXITY)
#include <ai/providers/perplexity/perplexity.hpp>
#endif
#if defined(AI_SDK_PROVIDER_COHERE)
#include <ai/providers/cohere/cohere.hpp>
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
            provider = ai::providers::anthropic::create_anthropic(std::move(o));
        }
#endif
#if defined(AI_SDK_PROVIDER_OPENAI)
        if (!provider && name == "openai") {
            ai::providers::openai::OpenAIOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
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
#endif
#if defined(AI_SDK_PROVIDER_GROQ)
        if (!provider && name == "groq") {
            ai::providers::groq::GroqOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::groq::create_groq(std::move(o));
        }
#endif
#if defined(AI_SDK_PROVIDER_XAI)
        if (!provider && name == "xai") {
            ai::providers::xai::XAIOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::xai::create_xai(std::move(o));
        }
#endif
#if defined(AI_SDK_PROVIDER_MISTRAL)
        if (!provider && name == "mistral") {
            ai::providers::mistral::MistralOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::mistral::create_mistral(std::move(o));
        }
#endif
#if defined(AI_SDK_PROVIDER_FIREWORKS)
        if (!provider && name == "fireworks") {
            ai::providers::fireworks::FireworksOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::fireworks::create_fireworks(std::move(o));
        }
#endif
#if defined(AI_SDK_PROVIDER_TOGETHERAI)
        if (!provider && name == "togetherai") {
            ai::providers::togetherai::TogetherAIOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::togetherai::create_togetherai(std::move(o));
        }
#endif
#if defined(AI_SDK_PROVIDER_PERPLEXITY)
        if (!provider && name == "perplexity") {
            ai::providers::perplexity::PerplexityOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::perplexity::create_perplexity(std::move(o));
        }
#endif
#if defined(AI_SDK_PROVIDER_COHERE)
        if (!provider && name == "cohere") {
            ai::providers::cohere::CohereOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::cohere::create_cohere(std::move(o));
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

const char* ai_sdk_version(void) {
    return "0.1.0";
}

} // extern "C"
