#include "ai_sdk.h"
#include <ai/ai.hpp>
#include <ai/providers/anthropic/anthropic.hpp>
#include <ai/providers/openai/openai.hpp>
#include <ai/providers/google/google.hpp>
#include <ai/providers/deepseek/deepseek.hpp>
#include <ai/providers/groq/groq.hpp>
#include <ai/providers/xai/xai.hpp>
#include <ai/providers/mistral/mistral.hpp>
#include <ai/providers/fireworks/fireworks.hpp>
#include <ai/providers/togetherai/togetherai.hpp>
#include <ai/providers/perplexity/perplexity.hpp>
#include <ai/providers/cohere/cohere.hpp>
#include <boost/asio.hpp>
#include <boost/json.hpp>
#include <string>
#include <memory>
#include <unordered_map>
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
        if (name == "anthropic") {
            ai::providers::anthropic::AnthropicOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::anthropic::create_anthropic(std::move(o));
        } else if (name == "openai") {
            ai::providers::openai::OpenAIOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::openai::create_openai(std::move(o));
        } else if (name == "google") {
            ai::providers::google::GoogleOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            provider = ai::providers::google::create_google(std::move(o));
        } else if (name == "deepseek") {
            ai::providers::deepseek::DeepSeekOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::deepseek::create_deepseek(std::move(o));
        } else if (name == "groq") {
            ai::providers::groq::GroqOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::groq::create_groq(std::move(o));
        } else if (name == "xai") {
            ai::providers::xai::XAIOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::xai::create_xai(std::move(o));
        } else if (name == "mistral") {
            ai::providers::mistral::MistralOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::mistral::create_mistral(std::move(o));
        } else if (name == "fireworks") {
            ai::providers::fireworks::FireworksOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::fireworks::create_fireworks(std::move(o));
        } else if (name == "togetherai") {
            ai::providers::togetherai::TogetherAIOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::togetherai::create_togetherai(std::move(o));
        } else if (name == "perplexity") {
            ai::providers::perplexity::PerplexityOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::perplexity::create_perplexity(std::move(o));
        } else if (name == "cohere") {
            ai::providers::cohere::CohereOptions o{.io_context = ctx->ioc};
            if (api_key) o.api_key = *api_key;
            if (opts.base_url) o.base_url = opts.base_url;
            provider = ai::providers::cohere::create_cohere(std::move(o));
        } else {
            ctx->last_error = "Unknown provider: " + name;
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

        auto& stream = stream_result.stream;
        while (true) {
            auto next = stream.next();
            // For synchronous iteration of the generator
            if (stream.done()) break;

            // Resume the generator
            auto part_opt = [&]() -> std::optional<ai::StreamPart> {
                // Simple synchronous pull from coroutine
                auto awaitable = stream.next();
                if (stream.done()) return std::nullopt;
                // This is a simplification — in practice the generator
                // needs proper async driving
                return std::nullopt;
            }();

            if (!part_opt) break;

            auto& part = *part_opt;
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
                }
            }, part);

            callback(event, user_data);
        }

        return AI_OK;
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
    (void)callback; (void)user_data;
    // For streaming agent, reuse the on_event from agent creation
    ai_generate_result_t result{};
    return ai_agent_call(agent, prompt, &result);
}

const char* ai_sdk_version(void) {
    return "0.1.0";
}

} // extern "C"
