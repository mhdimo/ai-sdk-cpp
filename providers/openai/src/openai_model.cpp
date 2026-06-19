#include <ai/providers/openai/openai_model.hpp>
#include <ai/providers/openai/openai_embedding.hpp>
#include <ai/providers/openai/openai_responses_model.hpp>
#include <ai/providers/openai/openai_batch.hpp>
#include <ai/providers/openai/openai.hpp>
#include <ai/stream/sse_parser.hpp>
#include <ai/util/json.hpp>
#include <ai/util/base64.hpp>
#include <ai/error/api_call_error.hpp>

namespace ai::providers::openai {

OpenAIProvider::OpenAIProvider(OpenAIOptions options)
    : options_(std::move(options))
    , http_client_(options_.http_client
        ? options_.http_client
        : std::make_shared<http::HttpClient>(options_.io_context)) {
    if (options_.api_key) {
        resolved_api_key_ = *options_.api_key;
    } else {
        const char* env_key = std::getenv("OPENAI_API_KEY");
        if (env_key) {
            resolved_api_key_ = env_key;
        }
    }
}

LanguageModelPtr OpenAIProvider::language_model(std::string_view model_id) {
    return std::make_shared<OpenAIChatLanguageModel>(std::string(model_id), shared_from_this());
}

std::shared_ptr<ai::batch::BatchProcessor> OpenAIProvider::batch_processor(std::string_view model_id) {
    return std::make_shared<OpenAIBatchProcessor>(*this, std::string(model_id));
}

LanguageModelPtr OpenAIProvider::responses_model(std::string_view model_id) {
    return std::make_shared<OpenAIResponsesModel>(std::string(model_id), *this);
}

EmbeddingModelPtr OpenAIProvider::embedding_model(std::string_view model_id) {
    return std::make_shared<OpenAIEmbeddingModel>(std::string(model_id), *this);
}

FileStoragePtr OpenAIProvider::file_storage() {
    return std::make_shared<OpenAIFileStorage>(*this);
}

http::Headers OpenAIProvider::auth_headers() const {
    http::Headers headers;
    if (!resolved_api_key_.empty()) {
        headers["Authorization"] = "Bearer " + resolved_api_key_;
    }
    if (options_.organization) {
        headers["OpenAI-Organization"] = *options_.organization;
    }
    if (options_.project) {
        headers["OpenAI-Project"] = *options_.project;
    }
    return headers;
}

std::string OpenAIProvider::chat_completions_url() const {
    return options_.base_url + "/chat/completions";
}

ProviderPtr create_openai(OpenAIOptions options) {
    return std::make_shared<OpenAIProvider>(std::move(options));
}

// --- OpenAIChatLanguageModel ---

namespace {

bool is_reasoning_model(std::string_view model_id) {
    // o1, o3, o4-mini are reasoning models
    if (model_id.starts_with("o1") || model_id.starts_with("o3") || model_id.starts_with("o4")) {
        return true;
    }
    // gpt-5 (but not gpt-5-chat) is a reasoning model
    if (model_id.starts_with("gpt-5") && model_id.find("-chat") == std::string_view::npos) {
        return true;
    }
    return false;
}

} // namespace

OpenAIChatLanguageModel::OpenAIChatLanguageModel(
    std::string model_id, std::shared_ptr<OpenAIProvider> provider
)
    : model_id_(std::move(model_id))
    , provider_(std::move(provider)) {}

boost::json::value OpenAIChatLanguageModel::build_request_body(
    const CallOptions& options, bool stream
) {
    boost::json::object body;
    body["model"] = model_id_;

    bool reasoning = is_reasoning_model(model_id_);

    if (stream) body["stream"] = true;

    // For reasoning models, use max_completion_tokens instead of max_tokens
    // and don't send temperature/top_p/frequency_penalty/presence_penalty
    if (reasoning) {
        if (options.max_output_tokens) {
            body["max_completion_tokens"] = *options.max_output_tokens;
        }
        // Add reasoning_effort based on options.reasoning
        if (options.reasoning) {
            auto& level = *options.reasoning;
            if (level == "low" || level == "minimal") {
                body["reasoning_effort"] = "low";
            } else if (level == "medium") {
                body["reasoning_effort"] = "medium";
            } else if (level == "high" || level == "xhigh") {
                body["reasoning_effort"] = "high";
            }
        }
    } else {
        if (options.max_output_tokens) body["max_tokens"] = *options.max_output_tokens;
        if (options.temperature) body["temperature"] = *options.temperature;
        if (options.top_p) body["top_p"] = *options.top_p;
        if (options.presence_penalty) body["presence_penalty"] = *options.presence_penalty;
        if (options.frequency_penalty) body["frequency_penalty"] = *options.frequency_penalty;
    }

    if (options.seed) body["seed"] = *options.seed;

    if (options.stop_sequences && !options.stop_sequences->empty()) {
        boost::json::array stops;
        for (auto& s : *options.stop_sequences) stops.push_back(boost::json::value(s));
        body["stop"] = std::move(stops);
    }

    // Messages
    boost::json::array messages;
    for (auto& msg : options.prompt) {
        std::visit([&](auto&& m) {
            using T = std::decay_t<decltype(m)>;
            if constexpr (std::is_same_v<T, SystemMessage>) {
                // For reasoning models, use "developer" role instead of "system"
                std::string role = reasoning ? "developer" : "system";
                messages.push_back(boost::json::object{
                    {"role", role}, {"content", m.content}
                });
            } else if constexpr (std::is_same_v<T, UserMessage>) {
                boost::json::array content;
                for (auto& part : m.content) {
                    std::visit([&](auto&& p) {
                        using P = std::decay_t<decltype(p)>;
                        if constexpr (std::is_same_v<P, TextPart>) {
                            content.push_back(boost::json::object{
                                {"type", "text"}, {"text", p.text}
                            });
                        } else if constexpr (std::is_same_v<P, FilePart>) {
                            // Serialize FilePart as image_url for OpenAI
                            std::visit([&](auto&& file_data) {
                                using FD = std::decay_t<decltype(file_data)>;
                                if constexpr (std::is_same_v<FD, DataFileData>) {
                                    std::string b64 = ai::util::base64_encode(file_data.data);
                                    std::string data_url = "data:" + p.media_type + ";base64," + b64;
                                    content.push_back(boost::json::object{
                                        {"type", "image_url"},
                                        {"image_url", boost::json::object{
                                            {"url", std::move(data_url)}
                                        }}
                                    });
                                } else if constexpr (std::is_same_v<FD, UrlFileData>) {
                                    content.push_back(boost::json::object{
                                        {"type", "image_url"},
                                        {"image_url", boost::json::object{
                                            {"url", file_data.url}
                                        }}
                                    });
                                }
                            }, p.data);
                        }
                    }, part);
                }
                messages.push_back(boost::json::object{
                    {"role", "user"}, {"content", std::move(content)}
                });
            } else if constexpr (std::is_same_v<T, AssistantMessage>) {
                boost::json::object msg_obj;
                msg_obj["role"] = "assistant";
                boost::json::array content;
                boost::json::array tool_calls_arr;

                for (auto& part : m.content) {
                    std::visit([&](auto&& p) {
                        using P = std::decay_t<decltype(p)>;
                        if constexpr (std::is_same_v<P, TextPart>) {
                            content.push_back(boost::json::object{
                                {"type", "text"}, {"text", p.text}
                            });
                        } else if constexpr (std::is_same_v<P, ToolCallPart>) {
                            tool_calls_arr.push_back(boost::json::object{
                                {"id", p.tool_call_id},
                                {"type", "function"},
                                {"function", boost::json::object{
                                    {"name", p.tool_name},
                                    {"arguments", boost::json::serialize(p.input)},
                                }},
                            });
                        }
                    }, part);
                }

                if (!content.empty()) msg_obj["content"] = std::move(content);
                if (!tool_calls_arr.empty()) msg_obj["tool_calls"] = std::move(tool_calls_arr);
                messages.push_back(std::move(msg_obj));
            } else if constexpr (std::is_same_v<T, ToolMessage>) {
                for (auto& part : m.content) {
                    std::string content_str;
                    std::visit([&](auto&& output) {
                        using O = std::decay_t<decltype(output)>;
                        if constexpr (std::is_same_v<O, TextOutput>) {
                            content_str = output.value;
                        } else if constexpr (std::is_same_v<O, JsonOutput>) {
                            content_str = boost::json::serialize(output.value);
                        } else if constexpr (std::is_same_v<O, ErrorTextOutput>) {
                            content_str = output.value;
                        } else if constexpr (std::is_same_v<O, ErrorJsonOutput>) {
                            content_str = boost::json::serialize(output.value);
                        }
                    }, part.output);

                    messages.push_back(boost::json::object{
                        {"role", "tool"},
                        {"tool_call_id", part.tool_call_id},
                        {"content", content_str},
                    });
                }
            }
        }, msg);
    }
    body["messages"] = std::move(messages);

    // Tools
    if (!options.tools.empty()) {
        boost::json::array tools;
        for (auto& tool : options.tools) {
            boost::json::object fn;
            fn["name"] = tool.name;
            if (tool.description) fn["description"] = *tool.description;
            fn["parameters"] = tool.input_schema.raw();
            if (tool.strict) fn["strict"] = true;

            tools.push_back(boost::json::object{
                {"type", "function"}, {"function", std::move(fn)}
            });
        }
        body["tools"] = std::move(tools);

        if (options.tool_choice) {
            std::visit([&](auto&& tc) {
                using TC = std::decay_t<decltype(tc)>;
                if constexpr (std::is_same_v<TC, ToolChoiceAuto>) {
                    body["tool_choice"] = "auto";
                } else if constexpr (std::is_same_v<TC, ToolChoiceNone>) {
                    body["tool_choice"] = "none";
                } else if constexpr (std::is_same_v<TC, ToolChoiceRequired>) {
                    body["tool_choice"] = "required";
                } else if constexpr (std::is_same_v<TC, ToolChoiceSpecific>) {
                    body["tool_choice"] = boost::json::object{
                        {"type", "function"},
                        {"function", boost::json::object{{"name", tc.tool_name}}}
                    };
                }
            }, *options.tool_choice);
        }
    }

    // Response format
    if (options.response_format && options.response_format->type == "json") {
        if (options.response_format->schema) {
            body["response_format"] = boost::json::object{
                {"type", "json_schema"},
                {"json_schema", boost::json::object{
                    {"name", options.response_format->name.value_or("response")},
                    {"schema", options.response_format->schema->raw()},
                    {"strict", true},
                }},
            };
        } else {
            body["response_format"] = boost::json::object{{"type", "json_object"}};
        }
    }

    return body;
}

GenerateResult OpenAIChatLanguageModel::parse_response(const boost::json::value& response) {
    GenerateResult result;
    auto& obj = response.as_object();

    auto choices_it = obj.find("choices");
    if (choices_it != obj.end() && choices_it->value().is_array()) {
        auto& choices = choices_it->value().as_array();
        if (!choices.empty()) {
            auto& choice = choices[0].as_object();

            // Finish reason
            if (auto fr = choice.find("finish_reason"); fr != choice.end() && fr->value().is_string()) {
                auto reason = std::string(fr->value().as_string());
                if (reason == "stop") result.finish_reason = FinishReason::Stop;
                else if (reason == "length") result.finish_reason = FinishReason::Length;
                else if (reason == "tool_calls") result.finish_reason = FinishReason::ToolCalls;
                else if (reason == "content_filter") result.finish_reason = FinishReason::ContentFilter;
                else result.finish_reason = FinishReason::Other;
            }

            // Message content
            auto msg_it = choice.find("message");
            if (msg_it != choice.end() && msg_it->value().is_object()) {
                auto& msg = msg_it->value().as_object();

                if (auto c = msg.find("content"); c != msg.end() && c->value().is_string()) {
                    result.content.push_back(TextContent{
                        .text = std::string(c->value().as_string())
                    });
                }

                if (auto tc = msg.find("tool_calls"); tc != msg.end() && tc->value().is_array()) {
                    for (auto& tool_call : tc->value().as_array()) {
                        if (!tool_call.is_object()) continue;
                        auto& tc_obj = tool_call.as_object();
                        auto fn_it = tc_obj.find("function");
                        if (fn_it == tc_obj.end() || !fn_it->value().is_object()) continue;
                        auto& fn = fn_it->value().as_object();
                        auto id_it = tc_obj.find("id");
                        auto name_it = fn.find("name");
                        auto args_it = fn.find("arguments");
                        if (!id_it || !id_it->value().is_string()) continue;
                        if (!name_it || !name_it->value().is_string()) continue;
                        result.content.push_back(ToolCallContent{
                            .tool_call_id = std::string(id_it->value().as_string()),
                            .tool_name = std::string(name_it->value().as_string()),
                            .input = (args_it != fn.end() && args_it->value().is_string())
                                ? std::string(args_it->value().as_string()) : "{}",
                        });
                    }
                }
            }
        }
    }

    // Usage
    auto usage_it = obj.find("usage");
    if (usage_it != obj.end() && usage_it->value().is_object()) {
        auto& usage = usage_it->value().as_object();
        if (auto it = usage.find("prompt_tokens"); it != usage.end()) {
            result.usage.input_tokens.total = static_cast<int>(it->value().as_int64());
        }
        if (auto it = usage.find("completion_tokens"); it != usage.end()) {
            result.usage.output_tokens.total = static_cast<int>(it->value().as_int64());
        }
    }

    // Response metadata
    result.response = ResponseMetadata{};
    if (auto id_it = obj.find("id"); id_it != obj.end() && id_it->value().is_string()) {
        result.response->id = std::string(id_it->value().as_string());
    }
    if (auto model_it = obj.find("model"); model_it != obj.end() && model_it->value().is_string()) {
        result.response->model_id = std::string(model_it->value().as_string());
    }

    return result;
}

Task<GenerateResult> OpenAIChatLanguageModel::do_generate(CallOptions options) {
    auto body = build_request_body(options, false);
    auto headers = provider_->auth_headers();
    auto url = provider_->chat_completions_url();

    auto response = co_await provider_->http_client().post_json(
        url, body, std::move(headers), options.cancel
    );

    auto parsed = ai::json::parse(response.body);
    co_return parse_response(parsed);
}

Task<StreamResult> OpenAIChatLanguageModel::do_stream(CallOptions options) {
    auto body = build_request_body(options, true);
    auto headers = provider_->auth_headers();
    auto url = provider_->chat_completions_url();

    auto response = co_await provider_->http_client().post_streaming(
        url, body, std::move(headers), options.cancel
    );

    stream::SseParser sse_parser;

    // Own the parser in the generator frame (see anthropic_model.cpp): parse()'s
    // coroutine references the parser by address and must outlive do_stream's
    // frame, which is destroyed once this Task completes while the stream is
    // drained later by the caller.
    auto stream = [](stream::SseParser parser,
                     AsyncGenerator<std::vector<uint8_t>> bytes) -> AsyncGenerator<StreamPart> {
        auto events = parser.parse(std::move(bytes));
        Usage usage{};
        FinishReason finish_reason = FinishReason::Stop;
        bool text_started = false;
        std::unordered_map<int, std::string> tool_call_ids;
        std::unordered_map<int, std::string> tool_call_names;
        std::unordered_map<int, bool> tool_call_started;

        co_yield StreamPart{StreamStart{}};

        while (auto event = co_await events.next()) {
            if (event->data == "[DONE]") continue;

            auto parsed = ai::json::safe_parse(event->data);
            if (!parsed || !parsed->is_object()) continue;

            auto& obj = parsed->as_object();
            auto choices_it = obj.find("choices");
            if (choices_it == obj.end() || !choices_it->value().is_array()) continue;

            auto& choices = choices_it->value().as_array();
            if (choices.empty()) continue;

            auto& choice = choices[0].as_object();

            // Finish reason
            if (auto fr = choice.find("finish_reason"); fr != choice.end() && fr->value().is_string()) {
                auto reason = std::string(fr->value().as_string());
                if (reason == "stop") finish_reason = FinishReason::Stop;
                else if (reason == "length") finish_reason = FinishReason::Length;
                else if (reason == "tool_calls") finish_reason = FinishReason::ToolCalls;
                else if (reason == "content_filter") finish_reason = FinishReason::ContentFilter;
            }

            auto delta_it = choice.find("delta");
            if (delta_it == choice.end() || !delta_it->value().is_object()) continue;
            auto& delta = delta_it->value().as_object();

            // Text content
            if (auto c = delta.find("content"); c != delta.end() && c->value().is_string()) {
                if (!text_started) {
                    co_yield StreamPart{TextStart{.id = "0"}};
                    text_started = true;
                }
                co_yield StreamPart{TextDelta{
                    .id = "0",
                    .delta = std::string(c->value().as_string()),
                }};
            }

            // Tool calls
            if (auto tc = delta.find("tool_calls"); tc != delta.end() && tc->value().is_array()) {
                for (auto& tool_call_delta : tc->value().as_array()) {
                    if (!tool_call_delta.is_object()) continue;
                    auto& tcd = tool_call_delta.as_object();
                    auto idx_it = tcd.find("index");
                    if (idx_it == tcd.end() || !idx_it->value().is_int64()) continue;
                    int index = static_cast<int>(idx_it->value().as_int64());

                    if (auto id_it = tcd.find("id"); id_it != tcd.end() && id_it->value().is_string()) {
                        tool_call_ids[index] = std::string(id_it->value().as_string());
                    }

                    if (auto fn = tcd.find("function"); fn != tcd.end() && fn->value().is_object()) {
                        auto& fn_obj = fn->value().as_object();
                        if (auto name = fn_obj.find("name"); name != fn_obj.end() && name->value().is_string()) {
                            tool_call_names[index] = std::string(name->value().as_string());
                        }

                        if (!tool_call_started[index] && tool_call_names.contains(index)) {
                            if (text_started) {
                                co_yield StreamPart{TextEnd{.id = "0"}};
                                text_started = false;
                            }
                            co_yield StreamPart{ToolInputStart{
                                .id = tool_call_ids[index],
                                .tool_name = tool_call_names[index],
                            }};
                            tool_call_started[index] = true;
                        }

                        if (auto args = fn_obj.find("arguments"); args != fn_obj.end() && args->value().is_string()) {
                            co_yield StreamPart{ToolInputDelta{
                                .id = tool_call_ids[index],
                                .delta = std::string(args->value().as_string()),
                            }};
                        }
                    }
                }
            }
        }

        if (text_started) {
            co_yield StreamPart{TextEnd{.id = "0"}};
        }
        for (auto& [index, started] : tool_call_started) {
            if (started) {
                co_yield StreamPart{ToolInputEnd{.id = tool_call_ids[index]}};
            }
        }

        co_yield StreamPart{FinishPart{.reason = finish_reason, .usage = usage}};
    }(std::move(sse_parser), std::move(response.body_stream));

    co_return StreamResult{
        .stream = std::move(stream),
        .response_headers = std::move(response.headers),
    };
}

} // namespace ai::providers::openai
