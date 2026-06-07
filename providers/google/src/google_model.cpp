#include <ai/providers/google/google_model.hpp>
#include <ai/providers/google/google.hpp>
#include <ai/stream/sse_parser.hpp>
#include <ai/util/json.hpp>
#include <ai/util/base64.hpp>
#include <ai/error/api_call_error.hpp>

namespace ai::providers::google {

GoogleLanguageModel::GoogleLanguageModel(
    std::string model_id, GoogleProvider& provider
)
    : model_id_(std::move(model_id))
    , provider_(provider) {}

boost::json::array GoogleLanguageModel::convert_contents(
    const Prompt& prompt, std::string& system_text
) {
    boost::json::array contents;

    for (auto& msg : prompt) {
        std::visit([&](auto&& m) {
            using T = std::decay_t<decltype(m)>;
            if constexpr (std::is_same_v<T, SystemMessage>) {
                system_text = m.content;
            } else if constexpr (std::is_same_v<T, UserMessage>) {
                boost::json::object content_obj;
                content_obj["role"] = "user";
                boost::json::array parts;
                for (auto& part : m.content) {
                    std::visit([&](auto&& p) {
                        using P = std::decay_t<decltype(p)>;
                        if constexpr (std::is_same_v<P, TextPart>) {
                            parts.push_back(boost::json::object{{"text", p.text}});
                        } else if constexpr (std::is_same_v<P, FilePart>) {
                            // Serialize FilePart as inlineData for Gemini
                            std::visit([&](auto&& file_data) {
                                using FD = std::decay_t<decltype(file_data)>;
                                if constexpr (std::is_same_v<FD, DataFileData>) {
                                    std::string b64 = ai::util::base64_encode(file_data.data);
                                    parts.push_back(boost::json::object{
                                        {"inlineData", boost::json::object{
                                            {"mimeType", p.media_type},
                                            {"data", std::move(b64)}
                                        }}
                                    });
                                } else if constexpr (std::is_same_v<FD, UrlFileData>) {
                                    parts.push_back(boost::json::object{
                                        {"fileData", boost::json::object{
                                            {"mimeType", p.media_type},
                                            {"fileUri", file_data.url}
                                        }}
                                    });
                                }
                            }, p.data);
                        }
                    }, part);
                }
                content_obj["parts"] = std::move(parts);
                contents.push_back(std::move(content_obj));
            } else if constexpr (std::is_same_v<T, AssistantMessage>) {
                boost::json::object content_obj;
                content_obj["role"] = "model";
                boost::json::array parts;
                for (auto& part : m.content) {
                    std::visit([&](auto&& p) {
                        using P = std::decay_t<decltype(p)>;
                        if constexpr (std::is_same_v<P, TextPart>) {
                            parts.push_back(boost::json::object{{"text", p.text}});
                        } else if constexpr (std::is_same_v<P, ToolCallPart>) {
                            boost::json::object fc;
                            fc["name"] = p.tool_name;
                            // Parse args from JSON value
                            if (p.input.is_object()) {
                                fc["args"] = p.input;
                            } else if (p.input.is_string()) {
                                auto parsed = ai::json::safe_parse(
                                    std::string(p.input.as_string()));
                                fc["args"] = parsed.value_or(boost::json::value(boost::json::object{}));
                            } else {
                                fc["args"] = boost::json::object{};
                            }
                            parts.push_back(boost::json::object{{"functionCall", std::move(fc)}});
                        }
                    }, part);
                }
                content_obj["parts"] = std::move(parts);
                contents.push_back(std::move(content_obj));
            } else if constexpr (std::is_same_v<T, ToolMessage>) {
                boost::json::object content_obj;
                content_obj["role"] = "user";
                boost::json::array parts;
                for (auto& part : m.content) {
                    boost::json::object fr;
                    fr["name"] = part.tool_name;
                    boost::json::object response;
                    std::visit([&](auto&& output) {
                        using O = std::decay_t<decltype(output)>;
                        if constexpr (std::is_same_v<O, TextOutput>) {
                            response["result"] = output.value;
                        } else if constexpr (std::is_same_v<O, JsonOutput>) {
                            response["result"] = output.value;
                        } else if constexpr (std::is_same_v<O, ErrorTextOutput>) {
                            response["error"] = output.value;
                        } else if constexpr (std::is_same_v<O, ErrorJsonOutput>) {
                            response["error"] = output.value;
                        } else {
                            response["result"] = "";
                        }
                    }, part.output);
                    fr["response"] = std::move(response);
                    parts.push_back(boost::json::object{{"functionResponse", std::move(fr)}});
                }
                content_obj["parts"] = std::move(parts);
                contents.push_back(std::move(content_obj));
            }
        }, msg);
    }

    return contents;
}

boost::json::array GoogleLanguageModel::convert_tools(
    const std::vector<FunctionTool>& tools
) {
    boost::json::array function_declarations;
    for (auto& tool : tools) {
        boost::json::object decl;
        decl["name"] = tool.name;
        if (tool.description) {
            decl["description"] = *tool.description;
        }
        decl["parameters"] = tool.input_schema.raw();
        function_declarations.push_back(std::move(decl));
    }
    return function_declarations;
}

boost::json::value GoogleLanguageModel::build_request_body(const CallOptions& options) {
    boost::json::object body;

    std::string system_text;
    body["contents"] = convert_contents(options.prompt, system_text);

    // System instruction
    if (!system_text.empty()) {
        body["systemInstruction"] = boost::json::object{
            {"parts", boost::json::array{boost::json::object{{"text", system_text}}}}
        };
    }

    // Generation config
    boost::json::object gen_config;
    if (options.max_output_tokens) gen_config["maxOutputTokens"] = *options.max_output_tokens;
    if (options.temperature) gen_config["temperature"] = *options.temperature;
    if (options.top_p) gen_config["topP"] = *options.top_p;
    if (options.top_k) gen_config["topK"] = *options.top_k;

    if (options.stop_sequences && !options.stop_sequences->empty()) {
        boost::json::array stops;
        for (auto& s : *options.stop_sequences) {
            stops.push_back(boost::json::value(s));
        }
        gen_config["stopSequences"] = std::move(stops);
    }

    // Response format / structured output
    if (options.response_format && options.response_format->type == "json") {
        gen_config["responseMimeType"] = "application/json";
        if (options.response_format->schema) {
            gen_config["responseSchema"] = options.response_format->schema->raw();
        }
    }

    if (!gen_config.empty()) {
        body["generationConfig"] = std::move(gen_config);
    }

    // Tools
    boost::json::array tools_array;
    bool has_tools = false;

    if (!options.tools.empty()) {
        auto declarations = convert_tools(options.tools);
        tools_array.push_back(
            boost::json::object{{"functionDeclarations", std::move(declarations)}}
        );
        has_tools = true;
    }

    // Google Search Grounding (provider_options["google"]["searchGrounding"])
    auto google_it = options.provider_options.find("google");
    if (google_it != options.provider_options.end() && google_it->value().is_object()) {
        auto& google_opts = google_it->value().as_object();

        auto search_it = google_opts.find("searchGrounding");
        if (search_it != google_opts.end()) {
            bool enabled = false;
            if (search_it->value().is_bool()) {
                enabled = search_it->value().as_bool();
            } else if (search_it->value().is_object()) {
                enabled = true;
            }
            if (enabled) {
                tools_array.push_back(
                    boost::json::object{{"googleSearchRetrieval", boost::json::object{}}}
                );
                has_tools = true;
            }
        }

        // Google Code Execution (provider_options["google"]["codeExecution"])
        auto code_it = google_opts.find("codeExecution");
        if (code_it != google_opts.end()) {
            bool enabled = false;
            if (code_it->value().is_bool()) {
                enabled = code_it->value().as_bool();
            } else if (code_it->value().is_object()) {
                enabled = true;
            }
            if (enabled) {
                tools_array.push_back(
                    boost::json::object{{"codeExecution", boost::json::object{}}}
                );
                has_tools = true;
            }
        }
    }

    if (has_tools) {
        body["tools"] = std::move(tools_array);

        // Tool config (tool_choice mapping)
        if (options.tool_choice) {
            std::visit([&](auto&& tc) {
                using TC = std::decay_t<decltype(tc)>;
                if constexpr (std::is_same_v<TC, ToolChoiceAuto>) {
                    body["toolConfig"] = boost::json::object{
                        {"functionCallingConfig", boost::json::object{{"mode", "AUTO"}}}
                    };
                } else if constexpr (std::is_same_v<TC, ToolChoiceNone>) {
                    body["toolConfig"] = boost::json::object{
                        {"functionCallingConfig", boost::json::object{{"mode", "NONE"}}}
                    };
                } else if constexpr (std::is_same_v<TC, ToolChoiceRequired>) {
                    body["toolConfig"] = boost::json::object{
                        {"functionCallingConfig", boost::json::object{{"mode", "ANY"}}}
                    };
                } else if constexpr (std::is_same_v<TC, ToolChoiceSpecific>) {
                    body["toolConfig"] = boost::json::object{
                        {"functionCallingConfig", boost::json::object{
                            {"mode", "ANY"},
                            {"allowedFunctionNames", boost::json::array{boost::json::value(tc.tool_name)}}
                        }}
                    };
                }
            }, *options.tool_choice);
        }
    }

    return body;
}

GenerateResult GoogleLanguageModel::parse_response(const boost::json::value& response) {
    GenerateResult result;
    auto& obj = response.as_object();

    // Parse candidates
    auto candidates_it = obj.find("candidates");
    if (candidates_it != obj.end() && candidates_it->value().is_array()) {
        auto& candidates = candidates_it->value().as_array();
        if (!candidates.empty()) {
            auto& candidate = candidates[0].as_object();

            // Finish reason
            if (auto fr = candidate.find("finishReason"); fr != candidate.end() && fr->value().is_string()) {
                auto reason = std::string(fr->value().as_string());
                if (reason == "STOP") {
                    result.finish_reason = FinishReason::Stop;
                } else if (reason == "MAX_TOKENS") {
                    result.finish_reason = FinishReason::Length;
                } else if (reason == "SAFETY") {
                    result.finish_reason = FinishReason::ContentFilter;
                } else {
                    result.finish_reason = FinishReason::Other;
                }
            }

            // Content parts
            auto content_it = candidate.find("content");
            if (content_it != candidate.end() && content_it->value().is_object()) {
                auto& content = content_it->value().as_object();
                auto parts_it = content.find("parts");
                if (parts_it != content.end() && parts_it->value().is_array()) {
                    bool has_function_calls = false;
                    for (auto& part : parts_it->value().as_array()) {
                        if (!part.is_object()) continue;
                        auto& part_obj = part.as_object();

                        // Text part
                        auto text_it = part_obj.find("text");
                        if (text_it != part_obj.end() && text_it->value().is_string()) {
                            result.content.push_back(TextContent{
                                .text = std::string(text_it->value().as_string())
                            });
                        }

                        // Function call part
                        auto fc_it = part_obj.find("functionCall");
                        if (fc_it != part_obj.end() && fc_it->value().is_object()) {
                            auto& fc = fc_it->value().as_object();
                            std::string name;
                            if (auto n = fc.find("name"); n != fc.end() && n->value().is_string()) {
                                name = std::string(n->value().as_string());
                            }
                            std::string args_str = "{}";
                            if (auto a = fc.find("args"); a != fc.end()) {
                                args_str = boost::json::serialize(a->value());
                            }
                            // Generate a synthetic tool call ID
                            std::string call_id = "call_" + name + "_" +
                                std::to_string(result.content.size());
                            result.content.push_back(ToolCallContent{
                                .tool_call_id = std::move(call_id),
                                .tool_name = std::move(name),
                                .input = std::move(args_str),
                            });
                            has_function_calls = true;
                        }
                    }

                    // If we got function calls, set finish reason to ToolCalls
                    if (has_function_calls && result.finish_reason == FinishReason::Stop) {
                        result.finish_reason = FinishReason::ToolCalls;
                    }
                }
            }
        }
    }

    // Parse usage metadata
    auto usage_it = obj.find("usageMetadata");
    if (usage_it != obj.end() && usage_it->value().is_object()) {
        auto& usage = usage_it->value().as_object();
        if (auto it = usage.find("promptTokenCount"); it != usage.end() && it->value().is_int64()) {
            result.usage.input_tokens.total = static_cast<int>(it->value().as_int64());
        }
        if (auto it = usage.find("candidatesTokenCount"); it != usage.end() && it->value().is_int64()) {
            result.usage.output_tokens.total = static_cast<int>(it->value().as_int64());
        }
    }

    // Response metadata
    result.response = ResponseMetadata{};
    result.response->model_id = model_id_;

    return result;
}

Task<GenerateResult> GoogleLanguageModel::do_generate(CallOptions options) {
    auto body = build_request_body(options);
    auto headers = provider_.content_headers();
    auto url = provider_.generate_url(model_id_);

    auto response = co_await provider_.http_client().post_json(
        url, body, std::move(headers), options.cancel
    );

    auto parsed = ai::json::parse(response.body);
    co_return parse_response(parsed);
}

Task<StreamResult> GoogleLanguageModel::do_stream(CallOptions options) {
    auto body = build_request_body(options);
    auto headers = provider_.content_headers();
    auto url = provider_.stream_url(model_id_);

    auto response = co_await provider_.http_client().post_streaming(
        url, body, std::move(headers), options.cancel
    );

    stream::SseParser sse_parser;
    auto sse_stream = sse_parser.parse(std::move(response.body_stream));

    auto stream = [model_id = model_id_](
        AsyncGenerator<stream::SseEvent> events
    ) -> AsyncGenerator<StreamPart> {
        Usage usage{};
        FinishReason finish_reason = FinishReason::Stop;
        bool text_started = false;
        int tool_call_index = 0;

        co_yield StreamPart{StreamStart{}};
        co_yield StreamPart{ResponseMetadataPart{
            .id = std::nullopt,
            .model_id = model_id,
        }};

        while (auto event = co_await events.next()) {
            if (event->data.empty() || event->data == "[DONE]") continue;

            auto parsed = ai::json::safe_parse(event->data);
            if (!parsed || !parsed->is_object()) continue;

            auto& obj = parsed->as_object();

            // Parse candidates from streaming chunk
            auto candidates_it = obj.find("candidates");
            if (candidates_it == obj.end() || !candidates_it->value().is_array()) continue;

            auto& candidates = candidates_it->value().as_array();
            if (candidates.empty()) continue;

            auto& candidate = candidates[0].as_object();

            // Check finish reason
            if (auto fr = candidate.find("finishReason"); fr != candidate.end() && fr->value().is_string()) {
                auto reason = std::string(fr->value().as_string());
                if (reason == "STOP") finish_reason = FinishReason::Stop;
                else if (reason == "MAX_TOKENS") finish_reason = FinishReason::Length;
                else if (reason == "SAFETY") finish_reason = FinishReason::ContentFilter;
            }

            // Parse content parts
            auto content_it = candidate.find("content");
            if (content_it == candidate.end() || !content_it->value().is_object()) continue;

            auto& content = content_it->value().as_object();
            auto parts_it = content.find("parts");
            if (parts_it == content.end() || !parts_it->value().is_array()) continue;

            for (auto& part : parts_it->value().as_array()) {
                if (!part.is_object()) continue;
                auto& part_obj = part.as_object();

                // Text content
                auto text_it = part_obj.find("text");
                if (text_it != part_obj.end() && text_it->value().is_string()) {
                    if (!text_started) {
                        co_yield StreamPart{TextStart{.id = "0"}};
                        text_started = true;
                    }
                    co_yield StreamPart{TextDelta{
                        .id = "0",
                        .delta = std::string(text_it->value().as_string()),
                    }};
                }

                // Function call
                auto fc_it = part_obj.find("functionCall");
                if (fc_it != part_obj.end() && fc_it->value().is_object()) {
                    if (text_started) {
                        co_yield StreamPart{TextEnd{.id = "0"}};
                        text_started = false;
                    }

                    auto& fc = fc_it->value().as_object();
                    std::string name;
                    if (auto n = fc.find("name"); n != fc.end() && n->value().is_string()) {
                        name = std::string(n->value().as_string());
                    }
                    std::string args_str = "{}";
                    if (auto a = fc.find("args"); a != fc.end()) {
                        args_str = boost::json::serialize(a->value());
                    }
                    std::string call_id = "call_" + name + "_" +
                        std::to_string(tool_call_index++);

                    co_yield StreamPart{ToolInputStart{
                        .id = call_id,
                        .tool_name = name,
                    }};
                    co_yield StreamPart{ToolInputDelta{
                        .id = call_id,
                        .delta = args_str,
                    }};
                    co_yield StreamPart{ToolInputEnd{.id = call_id}};

                    finish_reason = FinishReason::ToolCalls;
                }
            }

            // Parse usage from chunk
            auto usage_it = obj.find("usageMetadata");
            if (usage_it != obj.end() && usage_it->value().is_object()) {
                auto& u = usage_it->value().as_object();
                if (auto it = u.find("promptTokenCount"); it != u.end() && it->value().is_int64()) {
                    usage.input_tokens.total = static_cast<int>(it->value().as_int64());
                }
                if (auto it = u.find("candidatesTokenCount"); it != u.end() && it->value().is_int64()) {
                    usage.output_tokens.total = static_cast<int>(it->value().as_int64());
                }
            }
        }

        if (text_started) {
            co_yield StreamPart{TextEnd{.id = "0"}};
        }

        co_yield StreamPart{FinishPart{.reason = finish_reason, .usage = usage}};
    }(std::move(sse_stream));

    co_return StreamResult{
        .stream = std::move(stream),
        .response_headers = std::move(response.headers),
    };
}

} // namespace ai::providers::google
