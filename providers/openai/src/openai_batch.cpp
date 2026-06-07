#include <ai/providers/openai/openai_batch.hpp>
#include <ai/util/json.hpp>
#include <ai/error/api_call_error.hpp>
#include <ai/http/multipart.hpp>
#include <sstream>

namespace ai::providers::openai {

OpenAIBatchProcessor::OpenAIBatchProcessor(OpenAIProvider& provider, std::string model_id)
    : provider_(provider)
    , model_id_(std::move(model_id)) {}

boost::json::object OpenAIBatchProcessor::build_chat_body(const CallOptions& options) {
    // Build a chat completions request body similar to OpenAIChatLanguageModel
    boost::json::object body;
    body["model"] = model_id_;

    if (options.max_output_tokens) body["max_tokens"] = *options.max_output_tokens;
    if (options.temperature) body["temperature"] = *options.temperature;
    if (options.top_p) body["top_p"] = *options.top_p;
    if (options.presence_penalty) body["presence_penalty"] = *options.presence_penalty;
    if (options.frequency_penalty) body["frequency_penalty"] = *options.frequency_penalty;
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
                messages.push_back(boost::json::object{
                    {"role", "system"}, {"content", m.content}
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

ai::batch::BatchResponseItem OpenAIBatchProcessor::parse_response_line(
    const boost::json::value& line_val
) {
    ai::batch::BatchResponseItem item;
    auto& obj = line_val.as_object();

    if (auto cid = obj.find("custom_id"); cid != obj.end() && cid->value().is_string()) {
        item.custom_id = std::string(cid->value().as_string());
    }

    auto resp_it = obj.find("response");
    if (resp_it != obj.end() && resp_it->value().is_object()) {
        auto& resp = resp_it->value().as_object();

        auto status_it = resp.find("status_code");
        if (status_it != resp.end() && status_it->value().is_int64()) {
            int status_code = static_cast<int>(status_it->value().as_int64());
            if (status_code >= 400) {
                auto body_it = resp.find("body");
                if (body_it != resp.end()) {
                    item.error = boost::json::serialize(body_it->value());
                } else {
                    item.error = "Request failed with status " + std::to_string(status_code);
                }
                return item;
            }
        }

        auto body_it = resp.find("body");
        if (body_it != resp.end() && body_it->value().is_object()) {
            auto& body = body_it->value().as_object();
            // Parse as a chat completions response
            GenerateResult result;

            auto choices_it = body.find("choices");
            if (choices_it != body.end() && choices_it->value().is_array()) {
                auto& choices = choices_it->value().as_array();
                if (!choices.empty()) {
                    auto& choice = choices[0].as_object();

                    if (auto fr = choice.find("finish_reason"); fr != choice.end() && fr->value().is_string()) {
                        auto reason = std::string(fr->value().as_string());
                        if (reason == "stop") result.finish_reason = FinishReason::Stop;
                        else if (reason == "length") result.finish_reason = FinishReason::Length;
                        else if (reason == "tool_calls") result.finish_reason = FinishReason::ToolCalls;
                        else if (reason == "content_filter") result.finish_reason = FinishReason::ContentFilter;
                        else result.finish_reason = FinishReason::Other;
                    }

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
                                auto& tc_obj = tool_call.as_object();
                                auto& fn = tc_obj.at("function").as_object();
                                result.content.push_back(ToolCallContent{
                                    .tool_call_id = std::string(tc_obj.at("id").as_string()),
                                    .tool_name = std::string(fn.at("name").as_string()),
                                    .input = std::string(fn.at("arguments").as_string()),
                                });
                            }
                        }
                    }
                }
            }

            // Usage
            auto usage_it = body.find("usage");
            if (usage_it != body.end() && usage_it->value().is_object()) {
                auto& usage = usage_it->value().as_object();
                if (auto it = usage.find("prompt_tokens"); it != usage.end()) {
                    result.usage.input_tokens.total = static_cast<int>(it->value().as_int64());
                }
                if (auto it = usage.find("completion_tokens"); it != usage.end()) {
                    result.usage.output_tokens.total = static_cast<int>(it->value().as_int64());
                }
            }

            item.result = std::move(result);
        }
    }

    auto error_it = obj.find("error");
    if (error_it != obj.end() && error_it->value().is_object()) {
        item.error = boost::json::serialize(error_it->value());
    }

    return item;
}

ai::batch::BatchInfo OpenAIBatchProcessor::parse_batch_info(const boost::json::value& val) {
    ai::batch::BatchInfo info;
    auto& obj = val.as_object();

    if (auto it = obj.find("id"); it != obj.end() && it->value().is_string()) {
        info.id = std::string(it->value().as_string());
    }

    if (auto it = obj.find("status"); it != obj.end() && it->value().is_string()) {
        auto s = std::string(it->value().as_string());
        if (s == "validating") info.status = ai::batch::BatchStatus::Validating;
        else if (s == "in_progress") info.status = ai::batch::BatchStatus::InProgress;
        else if (s == "completed") info.status = ai::batch::BatchStatus::Completed;
        else if (s == "failed") info.status = ai::batch::BatchStatus::Failed;
        else if (s == "expired") info.status = ai::batch::BatchStatus::Expired;
        else if (s == "cancelling") info.status = ai::batch::BatchStatus::Cancelling;
        else if (s == "cancelled") info.status = ai::batch::BatchStatus::Cancelled;
        else info.status = ai::batch::BatchStatus::Failed;
    }

    // Request counts
    auto counts_it = obj.find("request_counts");
    if (counts_it != obj.end() && counts_it->value().is_object()) {
        auto& counts = counts_it->value().as_object();
        if (auto it = counts.find("total"); it != counts.end() && it->value().is_int64()) {
            info.total_requests = static_cast<int>(it->value().as_int64());
        }
        if (auto it = counts.find("completed"); it != counts.end() && it->value().is_int64()) {
            info.completed_requests = static_cast<int>(it->value().as_int64());
        }
        if (auto it = counts.find("failed"); it != counts.end() && it->value().is_int64()) {
            info.failed_requests = static_cast<int>(it->value().as_int64());
        }
    }

    if (auto it = obj.find("created_at"); it != obj.end() && it->value().is_int64()) {
        info.created_at = std::to_string(it->value().as_int64());
    }
    if (auto it = obj.find("completed_at"); it != obj.end() && it->value().is_int64()) {
        info.completed_at = std::to_string(it->value().as_int64());
    }

    return info;
}

Task<std::string> OpenAIBatchProcessor::submit(
    std::vector<ai::batch::BatchRequest> requests
) {
    // Step 1: Build JSONL content from requests
    std::ostringstream jsonl;
    for (auto& req : requests) {
        boost::json::object line;
        line["custom_id"] = req.custom_id;
        line["method"] = "POST";
        line["url"] = "/v1/chat/completions";
        line["body"] = build_chat_body(req.options);
        jsonl << boost::json::serialize(line) << "\n";
    }
    std::string jsonl_content = jsonl.str();

    // Step 2: Upload JSONL file via POST /v1/files (multipart)
    http::MultipartFormData form;
    form.add_field("purpose", "batch");
    form.add_file("file", "batch_input.jsonl", "application/jsonl",
        std::vector<uint8_t>(jsonl_content.begin(), jsonl_content.end()));

    std::string files_url = provider_.options().base_url + "/files";
    auto upload_response = co_await provider_.http_client().post_multipart(
        files_url, std::move(form), provider_.auth_headers()
    );

    auto upload_parsed = ai::json::parse(upload_response.body);
    auto file_id = ai::json::get_string(upload_parsed, "id");
    if (!file_id) {
        throw std::runtime_error("Failed to get file ID from upload response");
    }

    // Step 3: Create batch via POST /v1/batches
    boost::json::object batch_body;
    batch_body["input_file_id"] = *file_id;
    batch_body["endpoint"] = "/v1/chat/completions";
    batch_body["completion_window"] = "24h";

    std::string batches_url = provider_.options().base_url + "/batches";
    auto batch_response = co_await provider_.http_client().post_json(
        batches_url, batch_body, provider_.auth_headers()
    );

    auto batch_parsed = ai::json::parse(batch_response.body);
    auto batch_id = ai::json::get_string(batch_parsed, "id");
    if (!batch_id) {
        throw std::runtime_error("Failed to get batch ID from create response");
    }

    co_return *batch_id;
}

Task<ai::batch::BatchInfo> OpenAIBatchProcessor::status(std::string_view batch_id) {
    std::string url = provider_.options().base_url + "/batches/" + std::string(batch_id);

    auto response = co_await provider_.http_client().get(
        url, provider_.auth_headers()
    );

    auto parsed = ai::json::parse(response.body);
    co_return parse_batch_info(parsed);
}

Task<std::vector<ai::batch::BatchResponseItem>> OpenAIBatchProcessor::results(
    std::string_view batch_id
) {
    // First get the batch to find the output_file_id
    std::string batch_url = provider_.options().base_url + "/batches/" + std::string(batch_id);
    auto batch_response = co_await provider_.http_client().get(
        batch_url, provider_.auth_headers()
    );

    auto batch_parsed = ai::json::parse(batch_response.body);
    auto output_file_id = ai::json::get_string(batch_parsed, "output_file_id");
    if (!output_file_id) {
        throw std::runtime_error("Batch does not have an output file yet (not completed?)");
    }

    // Download the output file content
    std::string file_url = provider_.options().base_url + "/files/" + *output_file_id + "/content";
    auto file_response = co_await provider_.http_client().get(
        file_url, provider_.auth_headers()
    );

    // Parse JSONL response - each line is one result
    std::vector<ai::batch::BatchResponseItem> items;
    std::istringstream stream(file_response.body);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        auto line_parsed = ai::json::safe_parse(line);
        if (line_parsed && line_parsed->is_object()) {
            items.push_back(parse_response_line(*line_parsed));
        }
    }

    co_return items;
}

Task<void> OpenAIBatchProcessor::cancel(std::string_view batch_id) {
    std::string url = provider_.options().base_url + "/batches/" + std::string(batch_id) + "/cancel";

    co_await provider_.http_client().post_json(
        url, boost::json::object{}, provider_.auth_headers()
    );
}

Task<std::vector<ai::batch::BatchInfo>> OpenAIBatchProcessor::list(int limit) {
    std::string url = provider_.options().base_url + "/batches?limit=" + std::to_string(limit);

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

} // namespace ai::providers::openai
