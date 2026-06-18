#include <ai/providers/anthropic/anthropic_model.hpp>
#include <ai/providers/anthropic/anthropic.hpp>
#include <ai/stream/sse_parser.hpp>
#include <ai/util/json.hpp>
#include <ai/util/base64.hpp>
#include <ai/error/api_call_error.hpp>

namespace ai::providers::anthropic {

AnthropicLanguageModel::AnthropicLanguageModel(
    std::string model_id, AnthropicProvider& provider
)
    : model_id_(std::move(model_id))
    , provider_(provider) {}

int AnthropicLanguageModel::default_max_tokens() const {
    if (model_id_.find("opus") != std::string::npos) return 32000;
    if (model_id_.find("sonnet") != std::string::npos) return 64000;
    if (model_id_.find("haiku") != std::string::npos) return 64000;
    return 64000;
}

namespace {

// Map reasoning level string to Anthropic thinking budget_tokens
int reasoning_to_budget(std::string_view level) {
    if (level == "minimal" || level == "low") return 1024;
    if (level == "medium") return 10000;
    if (level == "high") return 50000;
    if (level == "xhigh") return 100000;
    return 0; // "none" or unrecognized
}

// Check if a content block has Anthropic cache_control in provider_options
void maybe_add_cache_control(boost::json::object& block, const ProviderOptions& provider_options) {
    auto anthropic_it = provider_options.find("anthropic");
    if (anthropic_it != provider_options.end() && anthropic_it->value().is_object()) {
        auto& anthropic_opts = anthropic_it->value().as_object();
        auto cc_it = anthropic_opts.find("cache_control");
        if (cc_it != anthropic_opts.end()) {
            block["cache_control"] = boost::json::object{{"type", "ephemeral"}};
        }
    }
}

} // namespace

boost::json::value AnthropicLanguageModel::build_request_body(
    const CallOptions& options, bool stream
) {
    boost::json::object body;
    body["model"] = model_id_;
    body["max_tokens"] = options.max_output_tokens.value_or(default_max_tokens());

    if (stream) {
        body["stream"] = true;
    }

    if (options.temperature) body["temperature"] = *options.temperature;
    if (options.top_p) body["top_p"] = *options.top_p;
    if (options.top_k) body["top_k"] = *options.top_k;

    if (options.stop_sequences && !options.stop_sequences->empty()) {
        boost::json::array stops;
        for (auto& s : *options.stop_sequences) {
            stops.push_back(boost::json::value(s));
        }
        body["stop_sequences"] = std::move(stops);
    }

    // Thinking/Reasoning controls
    if (options.reasoning && *options.reasoning != "none") {
        int budget = reasoning_to_budget(*options.reasoning);
        if (budget > 0) {
            body["thinking"] = boost::json::object{
                {"type", "enabled"},
                {"budget_tokens", budget}
            };
        }
    }

    // Convert messages
    boost::json::array messages;
    std::string system_text;
    boost::json::object system_provider_options;

    for (auto& msg : options.prompt) {
        std::visit([&](auto&& m) {
            using T = std::decay_t<decltype(m)>;
            if constexpr (std::is_same_v<T, SystemMessage>) {
                system_text = m.content;
                system_provider_options = m.provider_options;
            } else if constexpr (std::is_same_v<T, UserMessage>) {
                boost::json::object msg_obj;
                msg_obj["role"] = "user";
                boost::json::array content;
                for (auto& part : m.content) {
                    std::visit([&](auto&& p) {
                        using P = std::decay_t<decltype(p)>;
                        if constexpr (std::is_same_v<P, TextPart>) {
                            boost::json::object text_block;
                            text_block["type"] = "text";
                            text_block["text"] = p.text;
                            maybe_add_cache_control(text_block, p.provider_options);
                            content.push_back(std::move(text_block));
                        } else if constexpr (std::is_same_v<P, FilePart>) {
                            // Serialize FilePart based on media type
                            std::visit([&](auto&& file_data) {
                                using FD = std::decay_t<decltype(file_data)>;
                                if constexpr (std::is_same_v<FD, DataFileData>) {
                                    std::string b64 = ai::util::base64_encode(file_data.data);
                                    if (p.media_type == "application/pdf") {
                                        // PDF document
                                        boost::json::object doc_block;
                                        doc_block["type"] = "document";
                                        doc_block["source"] = boost::json::object{
                                            {"type", "base64"},
                                            {"media_type", p.media_type},
                                            {"data", std::move(b64)}
                                        };
                                        maybe_add_cache_control(doc_block, p.provider_options);
                                        content.push_back(std::move(doc_block));
                                    } else {
                                        // Image content
                                        boost::json::object img_block;
                                        img_block["type"] = "image";
                                        img_block["source"] = boost::json::object{
                                            {"type", "base64"},
                                            {"media_type", p.media_type},
                                            {"data", std::move(b64)}
                                        };
                                        maybe_add_cache_control(img_block, p.provider_options);
                                        content.push_back(std::move(img_block));
                                    }
                                } else if constexpr (std::is_same_v<FD, UrlFileData>) {
                                    // URL-based image
                                    boost::json::object img_block;
                                    img_block["type"] = "image";
                                    img_block["source"] = boost::json::object{
                                        {"type", "url"},
                                        {"url", file_data.url}
                                    };
                                    maybe_add_cache_control(img_block, p.provider_options);
                                    content.push_back(std::move(img_block));
                                }
                            }, p.data);
                        }
                    }, part);
                }
                msg_obj["content"] = std::move(content);
                // Cache control on message level
                maybe_add_cache_control(msg_obj, m.provider_options);
                messages.push_back(std::move(msg_obj));
            } else if constexpr (std::is_same_v<T, AssistantMessage>) {
                boost::json::object msg_obj;
                msg_obj["role"] = "assistant";
                boost::json::array content;
                for (auto& part : m.content) {
                    std::visit([&](auto&& p) {
                        using P = std::decay_t<decltype(p)>;
                        if constexpr (std::is_same_v<P, TextPart>) {
                            boost::json::object block;
                            block["type"] = "text";
                            block["text"] = p.text;
                            maybe_add_cache_control(block, p.provider_options);
                            content.push_back(std::move(block));
                        } else if constexpr (std::is_same_v<P, ToolCallPart>) {
                            boost::json::object block;
                            block["type"] = "tool_use";
                            block["id"] = p.tool_call_id;
                            block["name"] = p.tool_name;
                            block["input"] = p.input;
                            maybe_add_cache_control(block, p.provider_options);
                            content.push_back(std::move(block));
                        }
                    }, part);
                }
                msg_obj["content"] = std::move(content);
                maybe_add_cache_control(msg_obj, m.provider_options);
                messages.push_back(std::move(msg_obj));
            } else if constexpr (std::is_same_v<T, ToolMessage>) {
                boost::json::object msg_obj;
                msg_obj["role"] = "user";
                boost::json::array content;
                for (auto& part : m.content) {
                    boost::json::object block;
                    block["type"] = "tool_result";
                    block["tool_use_id"] = part.tool_call_id;
                    std::visit([&](auto&& output) {
                        using O = std::decay_t<decltype(output)>;
                        if constexpr (std::is_same_v<O, TextOutput>) {
                            block["content"] = output.value;
                        } else if constexpr (std::is_same_v<O, JsonOutput>) {
                            block["content"] = boost::json::serialize(output.value);
                        } else if constexpr (std::is_same_v<O, ErrorTextOutput>) {
                            block["content"] = output.value;
                            block["is_error"] = true;
                        } else if constexpr (std::is_same_v<O, ErrorJsonOutput>) {
                            block["content"] = boost::json::serialize(output.value);
                            block["is_error"] = true;
                        } else {
                            block["content"] = "";
                        }
                    }, part.output);
                    maybe_add_cache_control(block, part.provider_options);
                    content.push_back(std::move(block));
                }
                msg_obj["content"] = std::move(content);
                maybe_add_cache_control(msg_obj, m.provider_options);
                messages.push_back(std::move(msg_obj));
            }
        }, msg);
    }

    body["messages"] = std::move(messages);

    if (!system_text.empty()) {
        // Support cache control on system message
        auto anthropic_it = system_provider_options.find("anthropic");
        if (anthropic_it != system_provider_options.end() && anthropic_it->value().is_object()) {
            auto& anthropic_opts = anthropic_it->value().as_object();
            if (anthropic_opts.find("cache_control") != anthropic_opts.end()) {
                // Use array format for system with cache_control
                boost::json::array sys_arr;
                boost::json::object sys_block;
                sys_block["type"] = "text";
                sys_block["text"] = system_text;
                sys_block["cache_control"] = boost::json::object{{"type", "ephemeral"}};
                sys_arr.push_back(std::move(sys_block));
                body["system"] = std::move(sys_arr);
            } else {
                body["system"] = system_text;
            }
        } else {
            body["system"] = system_text;
        }
    }

    // Tools
    boost::json::array tools;
    bool has_tools = false;

    if (!options.tools.empty()) {
        for (auto& tool : options.tools) {
            boost::json::object tool_obj;
            tool_obj["name"] = tool.name;
            if (tool.description) {
                tool_obj["description"] = *tool.description;
            }
            tool_obj["input_schema"] = tool.input_schema.raw();
            tools.push_back(std::move(tool_obj));
        }
        has_tools = true;
    }

    // Built-in tools from provider_options
    auto anthropic_it = options.provider_options.find("anthropic");
    if (anthropic_it != options.provider_options.end() && anthropic_it->value().is_object()) {
        auto& anthropic_opts = anthropic_it->value().as_object();
        auto builtin_it = anthropic_opts.find("builtinTools");
        if (builtin_it != anthropic_opts.end() && builtin_it->value().is_array()) {
            for (auto& bt : builtin_it->value().as_array()) {
                if (!bt.is_object()) continue;
                auto& bt_obj = bt.as_object();
                boost::json::object tool_obj;

                // Type is required for built-in tools
                if (auto type_it = bt_obj.find("type"); type_it != bt_obj.end() && type_it->value().is_string()) {
                    tool_obj["type"] = type_it->value();
                }
                // Name is required
                if (auto name_it = bt_obj.find("name"); name_it != bt_obj.end() && name_it->value().is_string()) {
                    tool_obj["name"] = name_it->value();
                }
                // Merge any config properties directly into the tool object
                if (auto config_it = bt_obj.find("config"); config_it != bt_obj.end() && config_it->value().is_object()) {
                    for (auto& [key, val] : config_it->value().as_object()) {
                        tool_obj[key] = val;
                    }
                }

                tools.push_back(std::move(tool_obj));
                has_tools = true;
            }
        }
    }

    if (has_tools) {
        body["tools"] = std::move(tools);

        if (options.tool_choice) {
            std::visit([&](auto&& tc) {
                using TC = std::decay_t<decltype(tc)>;
                if constexpr (std::is_same_v<TC, ToolChoiceAuto>) {
                    body["tool_choice"] = boost::json::object{{"type", "auto"}};
                } else if constexpr (std::is_same_v<TC, ToolChoiceRequired>) {
                    body["tool_choice"] = boost::json::object{{"type", "any"}};
                } else if constexpr (std::is_same_v<TC, ToolChoiceSpecific>) {
                    body["tool_choice"] = boost::json::object{
                        {"type", "tool"}, {"name", tc.tool_name}
                    };
                }
            }, *options.tool_choice);
        }
    }

    // Structured output. Anthropic's Messages API has no top-level
    // response_format, so schema-conformant JSON is produced via the tool-use
    // pattern: a single tool whose input_schema is the requested schema, forced
    // via tool_choice. do_generate/do_stream then surface the tool input as text
    // so the provider-agnostic generate_object/stream_object path works.
    if (options.response_format && options.response_format->type == "json"
        && options.response_format->schema) {
        const std::string tool_name = options.response_format->name.value_or("json");

        boost::json::object tool_obj;
        tool_obj["name"] = tool_name;
        tool_obj["input_schema"] = options.response_format->schema->raw();
        tool_obj["description"] = options.response_format->description
            ? *options.response_format->description
            : std::string("Respond with a JSON object matching this schema.");
        tools.push_back(std::move(tool_obj));
        has_tools = true;

        body["tools"] = std::move(tools);
        body["tool_choice"] = boost::json::object{{"type", "tool"}, {"name", tool_name}};
    }

    return body;
}

GenerateResult AnthropicLanguageModel::parse_response(const boost::json::value& response) {
    GenerateResult result;

    auto& obj = response.as_object();

    // Parse content blocks
    if (auto content_it = obj.find("content"); content_it != obj.end() && content_it->value().is_array()) {
        for (auto& block : content_it->value().as_array()) {
            if (!block.is_object()) continue;
            auto& block_obj = block.as_object();
            auto type_it = block_obj.find("type");
            if (type_it == block_obj.end() || !type_it->value().is_string()) continue;
            auto type = std::string(type_it->value().as_string());

            if (type == "text") {
                auto text_it = block_obj.find("text");
                if (text_it != block_obj.end() && text_it->value().is_string()) {
                    result.content.push_back(TextContent{
                        .text = std::string(text_it->value().as_string())
                    });
                }
            } else if (type == "tool_use") {
                auto id_it = block_obj.find("id");
                auto name_it = block_obj.find("name");
                auto input_it = block_obj.find("input");
                if (id_it != block_obj.end() && id_it->value().is_string() &&
                    name_it != block_obj.end() && name_it->value().is_string()) {
                    std::string input_str = input_it != block_obj.end()
                        ? boost::json::serialize(input_it->value()) : "{}";
                    result.content.push_back(ToolCallContent{
                        .tool_call_id = std::string(id_it->value().as_string()),
                        .tool_name = std::string(name_it->value().as_string()),
                        .input = std::move(input_str),
                    });
                }
            } else if (type == "thinking") {
                auto thinking_it = block_obj.find("thinking");
                if (thinking_it != block_obj.end() && thinking_it->value().is_string()) {
                    result.content.push_back(ReasoningContent{
                        .text = std::string(thinking_it->value().as_string())
                    });
                }
            }
        }
    }

    // Parse finish reason
    auto stop_reason_it = obj.find("stop_reason");
    if (stop_reason_it != obj.end() && stop_reason_it->value().is_string()) {
        auto reason = std::string(stop_reason_it->value().as_string());
        if (reason == "end_turn" || reason == "stop_sequence") {
            result.finish_reason = FinishReason::Stop;
        } else if (reason == "max_tokens") {
            result.finish_reason = FinishReason::Length;
        } else if (reason == "tool_use") {
            result.finish_reason = FinishReason::ToolCalls;
        } else {
            result.finish_reason = FinishReason::Other;
        }
    }

    // Parse usage
    auto usage_it = obj.find("usage");
    if (usage_it != obj.end() && usage_it->value().is_object()) {
        auto& usage = usage_it->value().as_object();
        if (auto it = usage.find("input_tokens"); it != usage.end()) {
            result.usage.input_tokens.total = static_cast<int>(it->value().as_int64());
        }
        if (auto it = usage.find("output_tokens"); it != usage.end()) {
            result.usage.output_tokens.total = static_cast<int>(it->value().as_int64());
        }
        if (auto it = usage.find("cache_read_input_tokens"); it != usage.end() && !it->value().is_null()) {
            result.usage.input_tokens.cache_read = static_cast<int>(it->value().as_int64());
        }
        if (auto it = usage.find("cache_creation_input_tokens"); it != usage.end() && !it->value().is_null()) {
            result.usage.input_tokens.cache_write = static_cast<int>(it->value().as_int64());
        }
    }

    // Parse response metadata
    result.response = ResponseMetadata{};
    if (auto id_it = obj.find("id"); id_it != obj.end() && id_it->value().is_string()) {
        result.response->id = std::string(id_it->value().as_string());
    }
    if (auto model_it = obj.find("model"); model_it != obj.end() && model_it->value().is_string()) {
        result.response->model_id = std::string(model_it->value().as_string());
    }

    return result;
}

Task<GenerateResult> AnthropicLanguageModel::do_generate(CallOptions options) {
    const bool structured = options.response_format
        && options.response_format->type == "json"
        && options.response_format->schema;

    auto body = build_request_body(options, false);
    auto headers = provider_.auth_headers();
    auto url = provider_.messages_url();

    auto response = co_await provider_.http_client().post_json(
        url, body, std::move(headers), options.cancel
    );

    auto parsed = ai::json::parse(response.body);
    auto result = parse_response(parsed);

    // For structured output the JSON arrives in the forced tool_use block's
    // input; surface it as text so generate_object's text-parse path works.
    if (structured) {
        std::string json_input;
        for (auto& c : result.content) {
            if (auto* tc = std::get_if<ToolCallContent>(&c)) {
                json_input = tc->input;
                break;
            }
        }
        if (!json_input.empty()) {
            result.content.clear();
            result.content.push_back(TextContent{.text = std::move(json_input)});
            result.finish_reason = FinishReason::Stop;
        }
    }

    co_return result;
}

Task<StreamResult> AnthropicLanguageModel::do_stream(CallOptions options) {
    const bool structured = options.response_format
        && options.response_format->type == "json"
        && options.response_format->schema;

    auto body = build_request_body(options, true);
    auto headers = provider_.auth_headers();
    auto url = provider_.messages_url();

    auto response = co_await provider_.http_client().post_streaming(
        url, body, std::move(headers), options.cancel
    );

    stream::SseParser sse_parser;
    auto sse_stream = sse_parser.parse(std::move(response.body_stream));

    auto stream = [](stream::SseParser parser, AsyncGenerator<stream::SseEvent> events, bool structured) -> AsyncGenerator<StreamPart> {
        Usage usage{};
        FinishReason finish_reason = FinishReason::Stop;
        std::string current_tool_id;
        std::string current_tool_name;
        bool in_text = false;
        bool in_tool_input = false;

        while (auto event = co_await events.next()) {
            if (event->data == "[DONE]") continue;

            auto parsed = ai::json::safe_parse(event->data);
            if (!parsed) continue;

            auto& obj = parsed->as_object();
            auto type_it = obj.find("type");
            if (type_it == obj.end()) continue;
            auto type = std::string(type_it->value().as_string());

            if (type == "message_start") {
                auto msg_it = obj.find("message");
                if (msg_it != obj.end() && msg_it->value().is_object()) {
                    auto& msg = msg_it->value().as_object();
                    auto usage_it = msg.find("usage");
                    if (usage_it != msg.end() && usage_it->value().is_object()) {
                        auto& u = usage_it->value().as_object();
                        if (auto it = u.find("input_tokens"); it != u.end()) {
                            usage.input_tokens.total = static_cast<int>(it->value().as_int64());
                        }
                        if (auto it = u.find("cache_read_input_tokens"); it != u.end() && !it->value().is_null()) {
                            usage.input_tokens.cache_read = static_cast<int>(it->value().as_int64());
                        }
                        if (auto it = u.find("cache_creation_input_tokens"); it != u.end() && !it->value().is_null()) {
                            usage.input_tokens.cache_write = static_cast<int>(it->value().as_int64());
                        }
                    }

                    std::optional<std::string> id, model_id;
                    if (auto it = msg.find("id"); it != msg.end() && it->value().is_string())
                        id = std::string(it->value().as_string());
                    if (auto it = msg.find("model"); it != msg.end() && it->value().is_string())
                        model_id = std::string(it->value().as_string());

                    co_yield StreamPart{ResponseMetadataPart{.id = id, .model_id = model_id}};
                }
                co_yield StreamPart{StreamStart{}};
            } else if (type == "content_block_start") {
                auto block_it = obj.find("content_block");
                if (block_it != obj.end() && block_it->value().is_object()) {
                    auto& block = block_it->value().as_object();
                    auto bt_it = block.find("type");
                    if (bt_it == block.end() || !bt_it->value().is_string()) continue;
                    auto block_type = std::string(bt_it->value().as_string());

                    if (block_type == "text") {
                        in_text = true;
                        co_yield StreamPart{TextStart{.id = "0"}};
                    } else if (block_type == "tool_use") {
                        auto id_it = block.find("id");
                        auto name_it = block.find("name");
                        if (id_it != block.end() && id_it->value().is_string())
                            current_tool_id = std::string(id_it->value().as_string());
                        if (name_it != block.end() && name_it->value().is_string())
                            current_tool_name = std::string(name_it->value().as_string());
                        if (structured) {
                            // Map the forced JSON tool's input onto the text
                            // channel so stream_object can accumulate it.
                            in_text = true;
                            co_yield StreamPart{TextStart{.id = "0"}};
                        } else {
                            in_tool_input = true;
                            co_yield StreamPart{ToolInputStart{
                                .id = current_tool_id,
                                .tool_name = current_tool_name,
                            }};
                        }
                    } else if (block_type == "thinking") {
                        co_yield StreamPart{ReasoningStart{.id = "0"}};
                    }
                }
            } else if (type == "content_block_delta") {
                auto delta_it = obj.find("delta");
                if (delta_it != obj.end() && delta_it->value().is_object()) {
                    auto& delta = delta_it->value().as_object();
                    auto dt_it = delta.find("type");
                    if (dt_it == delta.end() || !dt_it->value().is_string()) continue;
                    auto delta_type = std::string(dt_it->value().as_string());

                    if (delta_type == "text_delta") {
                        auto t_it = delta.find("text");
                        if (t_it != delta.end() && t_it->value().is_string()) {
                            co_yield StreamPart{TextDelta{.id = "0", .delta = std::string(t_it->value().as_string())}};
                        }
                    } else if (delta_type == "input_json_delta") {
                        auto pj_it = delta.find("partial_json");
                        if (pj_it != delta.end() && pj_it->value().is_string()) {
                            auto pj = std::string(pj_it->value().as_string());
                            if (structured) {
                                co_yield StreamPart{TextDelta{.id = "0", .delta = std::move(pj)}};
                            } else {
                                co_yield StreamPart{ToolInputDelta{
                                    .id = current_tool_id,
                                    .delta = std::move(pj),
                                }};
                            }
                        }
                    } else if (delta_type == "thinking_delta") {
                        auto th_it = delta.find("thinking");
                        if (th_it != delta.end() && th_it->value().is_string()) {
                            co_yield StreamPart{ReasoningDelta{.id = "0", .delta = std::string(th_it->value().as_string())}};
                        }
                    }
                }
            } else if (type == "content_block_stop") {
                if (in_text) {
                    co_yield StreamPart{TextEnd{.id = "0"}};
                    in_text = false;
                } else if (in_tool_input) {
                    co_yield StreamPart{ToolInputEnd{.id = current_tool_id}};
                    in_tool_input = false;
                } else {
                    co_yield StreamPart{ReasoningEnd{.id = "0"}};
                }
            } else if (type == "message_delta") {
                auto delta_it = obj.find("delta");
                if (delta_it != obj.end() && delta_it->value().is_object()) {
                    auto& delta = delta_it->value().as_object();
                    if (auto sr = delta.find("stop_reason"); sr != delta.end() && sr->value().is_string()) {
                        auto reason = std::string(sr->value().as_string());
                        if (reason == "end_turn" || reason == "stop_sequence")
                            finish_reason = FinishReason::Stop;
                        else if (reason == "max_tokens")
                            finish_reason = FinishReason::Length;
                        else if (reason == "tool_use")
                            finish_reason = FinishReason::ToolCalls;
                    }
                }
                auto usage_delta_it = obj.find("usage");
                if (usage_delta_it != obj.end() && usage_delta_it->value().is_object()) {
                    auto& u = usage_delta_it->value().as_object();
                    if (auto it = u.find("output_tokens"); it != u.end()) {
                        usage.output_tokens.total = static_cast<int>(it->value().as_int64());
                    }
                    if (auto it = u.find("cache_read_input_tokens"); it != u.end() && !it->value().is_null()) {
                        usage.input_tokens.cache_read = static_cast<int>(it->value().as_int64());
                    }
                    if (auto it = u.find("cache_creation_input_tokens"); it != u.end() && !it->value().is_null()) {
                        usage.input_tokens.cache_write = static_cast<int>(it->value().as_int64());
                    }
                }
            } else if (type == "message_stop") {
                co_yield StreamPart{FinishPart{.reason = finish_reason, .usage = usage}};
            } else if (type == "error") {
                auto err_it = obj.find("error");
                std::string msg = "Unknown streaming error";
                if (err_it != obj.end() && err_it->value().is_object()) {
                    auto& err = err_it->value().as_object();
                    if (auto m = err.find("message"); m != err.end() && m->value().is_string()) {
                        msg = std::string(m->value().as_string());
                    }
                }
                co_yield StreamPart{ErrorPart{.message = std::move(msg)}};
            }
        }
    }(std::move(sse_parser), std::move(sse_stream), structured);

    co_return StreamResult{
        .stream = std::move(stream),
        .response_headers = std::move(response.headers),
    };
}

} // namespace ai::providers::anthropic
