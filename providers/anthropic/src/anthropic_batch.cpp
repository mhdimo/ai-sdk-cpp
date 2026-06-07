#include <ai/providers/anthropic/anthropic_batch.hpp>
#include <ai/providers/anthropic/anthropic.hpp>
#include <ai/util/json.hpp>
#include <ai/error/api_call_error.hpp>
#include <sstream>

namespace ai::providers::anthropic {

AnthropicBatchProcessor::AnthropicBatchProcessor(
    AnthropicProvider& provider, std::string model_id
)
    : provider_(provider)
    , model_id_(std::move(model_id)) {}

boost::json::object AnthropicBatchProcessor::build_messages_params(const CallOptions& options) {
    boost::json::object params;
    params["model"] = model_id_;
    params["max_tokens"] = options.max_output_tokens.value_or(4096);

    if (options.temperature) params["temperature"] = *options.temperature;
    if (options.top_p) params["top_p"] = *options.top_p;
    if (options.top_k) params["top_k"] = *options.top_k;

    if (options.stop_sequences && !options.stop_sequences->empty()) {
        boost::json::array stops;
        for (auto& s : *options.stop_sequences) {
            stops.push_back(boost::json::value(s));
        }
        params["stop_sequences"] = std::move(stops);
    }

    // Convert messages
    boost::json::array messages;
    std::string system_text;

    for (auto& msg : options.prompt) {
        std::visit([&](auto&& m) {
            using T = std::decay_t<decltype(m)>;
            if constexpr (std::is_same_v<T, SystemMessage>) {
                system_text = m.content;
            } else if constexpr (std::is_same_v<T, UserMessage>) {
                boost::json::object msg_obj;
                msg_obj["role"] = "user";
                boost::json::array content;
                for (auto& part : m.content) {
                    std::visit([&](auto&& p) {
                        using P = std::decay_t<decltype(p)>;
                        if constexpr (std::is_same_v<P, TextPart>) {
                            content.push_back(boost::json::object{
                                {"type", "text"}, {"text", p.text}
                            });
                        }
                    }, part);
                }
                msg_obj["content"] = std::move(content);
                messages.push_back(std::move(msg_obj));
            } else if constexpr (std::is_same_v<T, AssistantMessage>) {
                boost::json::object msg_obj;
                msg_obj["role"] = "assistant";
                boost::json::array content;
                for (auto& part : m.content) {
                    std::visit([&](auto&& p) {
                        using P = std::decay_t<decltype(p)>;
                        if constexpr (std::is_same_v<P, TextPart>) {
                            content.push_back(boost::json::object{
                                {"type", "text"}, {"text", p.text}
                            });
                        } else if constexpr (std::is_same_v<P, ToolCallPart>) {
                            content.push_back(boost::json::object{
                                {"type", "tool_use"},
                                {"id", p.tool_call_id},
                                {"name", p.tool_name},
                                {"input", p.input},
                            });
                        }
                    }, part);
                }
                msg_obj["content"] = std::move(content);
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
                    content.push_back(std::move(block));
                }
                msg_obj["content"] = std::move(content);
                messages.push_back(std::move(msg_obj));
            }
        }, msg);
    }

    params["messages"] = std::move(messages);

    if (!system_text.empty()) {
        params["system"] = system_text;
    }

    // Tools
    if (!options.tools.empty()) {
        boost::json::array tools;
        for (auto& tool : options.tools) {
            boost::json::object tool_obj;
            tool_obj["name"] = tool.name;
            if (tool.description) {
                tool_obj["description"] = *tool.description;
            }
            tool_obj["input_schema"] = tool.input_schema.raw();
            tools.push_back(std::move(tool_obj));
        }
        params["tools"] = std::move(tools);

        if (options.tool_choice) {
            std::visit([&](auto&& tc) {
                using TC = std::decay_t<decltype(tc)>;
                if constexpr (std::is_same_v<TC, ToolChoiceAuto>) {
                    params["tool_choice"] = boost::json::object{{"type", "auto"}};
                } else if constexpr (std::is_same_v<TC, ToolChoiceRequired>) {
                    params["tool_choice"] = boost::json::object{{"type", "any"}};
                } else if constexpr (std::is_same_v<TC, ToolChoiceSpecific>) {
                    params["tool_choice"] = boost::json::object{
                        {"type", "tool"}, {"name", tc.tool_name}
                    };
                }
            }, *options.tool_choice);
        }
    }

    return params;
}

ai::batch::BatchResponseItem AnthropicBatchProcessor::parse_result_line(
    const boost::json::value& line_val
) {
    ai::batch::BatchResponseItem item;
    auto& obj = line_val.as_object();

    if (auto cid = obj.find("custom_id"); cid != obj.end() && cid->value().is_string()) {
        item.custom_id = std::string(cid->value().as_string());
    }

    auto result_it = obj.find("result");
    if (result_it == obj.end() || !result_it->value().is_object()) {
        item.error = "Missing result in response line";
        return item;
    }

    auto& result_obj = result_it->value().as_object();
    auto type_it = result_obj.find("type");
    if (type_it == result_obj.end() || !type_it->value().is_string()) {
        item.error = "Missing type in result";
        return item;
    }

    auto type = std::string(type_it->value().as_string());

    if (type == "errored") {
        auto error_it = result_obj.find("error");
        if (error_it != result_obj.end()) {
            item.error = boost::json::serialize(error_it->value());
        } else {
            item.error = "Request errored";
        }
        return item;
    }

    if (type == "canceled") {
        item.error = "Request was canceled";
        return item;
    }

    if (type == "expired") {
        item.error = "Request expired";
        return item;
    }

    // type == "succeeded"
    auto message_it = result_obj.find("message");
    if (message_it == result_obj.end() || !message_it->value().is_object()) {
        item.error = "Missing message in succeeded result";
        return item;
    }

    auto& msg_obj = message_it->value().as_object();
    GenerateResult result;

    // Parse content blocks
    if (auto content_it = msg_obj.find("content"); content_it != msg_obj.end() && content_it->value().is_array()) {
        for (auto& block : content_it->value().as_array()) {
            if (!block.is_object()) continue;
            auto& block_obj = block.as_object();
            auto block_type_it = block_obj.find("type");
            if (block_type_it == block_obj.end() || !block_type_it->value().is_string()) continue;

            auto block_type = std::string(block_type_it->value().as_string());

            if (block_type == "text") {
                if (auto t = block_obj.find("text"); t != block_obj.end() && t->value().is_string()) {
                    result.content.push_back(TextContent{
                        .text = std::string(t->value().as_string())
                    });
                }
            } else if (block_type == "tool_use") {
                result.content.push_back(ToolCallContent{
                    .tool_call_id = std::string(block_obj.at("id").as_string()),
                    .tool_name = std::string(block_obj.at("name").as_string()),
                    .input = boost::json::serialize(block_obj.at("input")),
                });
            } else if (block_type == "thinking") {
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
    if (auto sr = msg_obj.find("stop_reason"); sr != msg_obj.end() && sr->value().is_string()) {
        auto reason = std::string(sr->value().as_string());
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
    if (auto usage_it = msg_obj.find("usage"); usage_it != msg_obj.end() && usage_it->value().is_object()) {
        auto& usage = usage_it->value().as_object();
        if (auto it = usage.find("input_tokens"); it != usage.end() && it->value().is_int64()) {
            result.usage.input_tokens.total = static_cast<int>(it->value().as_int64());
        }
        if (auto it = usage.find("output_tokens"); it != usage.end() && it->value().is_int64()) {
            result.usage.output_tokens.total = static_cast<int>(it->value().as_int64());
        }
    }

    item.result = std::move(result);
    return item;
}

ai::batch::BatchInfo AnthropicBatchProcessor::parse_batch_info(const boost::json::value& val) {
    ai::batch::BatchInfo info;
    auto& obj = val.as_object();

    if (auto it = obj.find("id"); it != obj.end() && it->value().is_string()) {
        info.id = std::string(it->value().as_string());
    }

    if (auto it = obj.find("processing_status"); it != obj.end() && it->value().is_string()) {
        auto s = std::string(it->value().as_string());
        if (s == "in_progress") info.status = ai::batch::BatchStatus::InProgress;
        else if (s == "ended") info.status = ai::batch::BatchStatus::Completed;
        else if (s == "canceling") info.status = ai::batch::BatchStatus::Cancelling;
        else info.status = ai::batch::BatchStatus::InProgress;
    }

    // Request counts
    auto counts_it = obj.find("request_counts");
    if (counts_it != obj.end() && counts_it->value().is_object()) {
        auto& counts = counts_it->value().as_object();
        if (auto it = counts.find("processing"); it != counts.end() && it->value().is_int64()) {
            info.total_requests += static_cast<int>(it->value().as_int64());
        }
        if (auto it = counts.find("succeeded"); it != counts.end() && it->value().is_int64()) {
            info.completed_requests = static_cast<int>(it->value().as_int64());
            info.total_requests += info.completed_requests;
        }
        if (auto it = counts.find("errored"); it != counts.end() && it->value().is_int64()) {
            info.failed_requests = static_cast<int>(it->value().as_int64());
            info.total_requests += info.failed_requests;
        }
        if (auto it = counts.find("canceled"); it != counts.end() && it->value().is_int64()) {
            info.total_requests += static_cast<int>(it->value().as_int64());
        }
        if (auto it = counts.find("expired"); it != counts.end() && it->value().is_int64()) {
            info.total_requests += static_cast<int>(it->value().as_int64());
        }
    }

    if (auto it = obj.find("created_at"); it != obj.end() && it->value().is_string()) {
        info.created_at = std::string(it->value().as_string());
    }
    if (auto it = obj.find("ended_at"); it != obj.end() && it->value().is_string()) {
        info.completed_at = std::string(it->value().as_string());
    }

    return info;
}

Task<std::string> AnthropicBatchProcessor::submit(
    std::vector<ai::batch::BatchRequest> requests
) {
    // Build request body with inline requests
    boost::json::object body;
    boost::json::array request_array;

    for (auto& req : requests) {
        boost::json::object request_obj;
        request_obj["custom_id"] = req.custom_id;
        request_obj["params"] = build_messages_params(req.options);
        request_array.push_back(std::move(request_obj));
    }

    body["requests"] = std::move(request_array);

    std::string url = provider_.options().base_url + "/v1/messages/batches";
    auto response = co_await provider_.http_client().post_json(
        url, body, provider_.auth_headers()
    );

    auto parsed = ai::json::parse(response.body);
    auto batch_id = ai::json::get_string(parsed, "id");
    if (!batch_id) {
        throw std::runtime_error("Failed to get batch ID from create response");
    }

    co_return *batch_id;
}

Task<ai::batch::BatchInfo> AnthropicBatchProcessor::status(std::string_view batch_id) {
    std::string url = provider_.options().base_url + "/v1/messages/batches/" + std::string(batch_id);

    auto response = co_await provider_.http_client().get(
        url, provider_.auth_headers()
    );

    auto parsed = ai::json::parse(response.body);
    co_return parse_batch_info(parsed);
}

Task<std::vector<ai::batch::BatchResponseItem>> AnthropicBatchProcessor::results(
    std::string_view batch_id
) {
    std::string url = provider_.options().base_url + "/v1/messages/batches/"
        + std::string(batch_id) + "/results";

    auto response = co_await provider_.http_client().get(
        url, provider_.auth_headers()
    );

    // Parse JSONL response - each line is one result
    std::vector<ai::batch::BatchResponseItem> items;
    std::istringstream stream(response.body);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        auto line_parsed = ai::json::safe_parse(line);
        if (line_parsed && line_parsed->is_object()) {
            items.push_back(parse_result_line(*line_parsed));
        }
    }

    co_return items;
}

Task<void> AnthropicBatchProcessor::cancel(std::string_view batch_id) {
    std::string url = provider_.options().base_url + "/v1/messages/batches/"
        + std::string(batch_id) + "/cancel";

    co_await provider_.http_client().post_json(
        url, boost::json::object{}, provider_.auth_headers()
    );
}

Task<std::vector<ai::batch::BatchInfo>> AnthropicBatchProcessor::list(int limit) {
    std::string url = provider_.options().base_url + "/v1/messages/batches?limit="
        + std::to_string(limit);

    auto response = co_await provider_.http_client().get(
        url, provider_.auth_headers()
    );

    auto parsed = ai::json::parse(response.body);
    std::vector<ai::batch::BatchInfo> batches;

    if (parsed.is_object()) {
        auto data_it = parsed.as_object().find("data");
        if (data_it != parsed.as_object().end() && data_it->value().is_array()) {
            for (auto& item : data_it->value().as_array()) {
                batches.push_back(parse_batch_info(item));
            }
        }
    }

    co_return batches;
}

} // namespace ai::providers::anthropic
