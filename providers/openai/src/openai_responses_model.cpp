#include <ai/providers/openai/openai_responses_model.hpp>
#include <ai/providers/openai/openai.hpp>
#include <ai/stream/stream_part.hpp>
#include <ai/stream/async_generator.hpp>
#include <ai/http/client.hpp>
#include <boost/json.hpp>

namespace ai::providers::openai {

namespace json = boost::json;

OpenAIResponsesModel::OpenAIResponsesModel(std::string model_id, OpenAIProvider& provider)
    : model_id_(std::move(model_id)), provider_(provider) {}

json::object OpenAIResponsesModel::build_request_body(const CallOptions& options, bool stream) {
    json::object body;
    body["model"] = model_id_;

    // Convert prompt to input
    json::array input;
    std::string instructions;

    for (auto& msg : options.prompt) {
        if (auto* sys = std::get_if<SystemMessage>(&msg)) {
            instructions = sys->content;
        } else if (auto* user = std::get_if<UserMessage>(&msg)) {
            std::string text;
            for (auto& part : user->content) {
                if (auto* t = std::get_if<TextPart>(&part)) {
                    text += t->text;
                }
            }
            input.push_back(json::object{{"role", "user"}, {"content", text}});
        } else if (auto* asst = std::get_if<AssistantMessage>(&msg)) {
            std::string text;
            for (auto& part : asst->content) {
                if (auto* t = std::get_if<TextPart>(&part)) {
                    text += t->text;
                }
            }
            input.push_back(json::object{{"role", "assistant"}, {"content", text}});
        }
    }

    if (!instructions.empty()) body["instructions"] = instructions;
    body["input"] = std::move(input);

    if (options.max_output_tokens) body["max_output_tokens"] = *options.max_output_tokens;
    if (options.temperature) body["temperature"] = *options.temperature;
    if (options.top_p) body["top_p"] = *options.top_p;

    // Tools
    if (!options.tools.empty()) {
        json::array tools;
        for (auto& t : options.tools) {
            json::object tool;
            tool["type"] = "function";
            tool["name"] = t.name;
            if (t.description) tool["description"] = *t.description;
            tool["parameters"] = json::parse(t.input_schema.to_string());
            if (t.strict) tool["strict"] = true;
            tools.push_back(std::move(tool));
        }
        body["tools"] = std::move(tools);
    }

    if (options.tool_choice) {
        std::visit([&](auto& choice) {
            using T = std::decay_t<decltype(choice)>;
            if constexpr (std::is_same_v<T, ToolChoiceAuto>) {
                body["tool_choice"] = "auto";
            } else if constexpr (std::is_same_v<T, ToolChoiceNone>) {
                body["tool_choice"] = "none";
            } else if constexpr (std::is_same_v<T, ToolChoiceRequired>) {
                body["tool_choice"] = "required";
            } else if constexpr (std::is_same_v<T, ToolChoiceSpecific>) {
                body["tool_choice"] = json::object{
                    {"type", "function"},
                    {"name", json::value(choice.tool_name)}
                };
            }
        }, *options.tool_choice);
    }

    if (stream) body["stream"] = true;

    return body;
}

GenerateResult OpenAIResponsesModel::parse_response(const json::value& response) {
    GenerateResult result;
    if (!response.is_object()) return result;
    auto& obj = response.as_object();

    auto output_it = obj.find("output");
    if (output_it != obj.end() && output_it->value().is_array()) {
        for (auto& item : output_it->value().as_array()) {
            if (!item.is_object()) continue;
            auto& item_obj = item.as_object();
            auto type_it = item_obj.find("type");
            if (type_it == item_obj.end() || !type_it->value().is_string()) continue;
            auto type = std::string(type_it->value().as_string());

            if (type == "message") {
                auto content_it = item_obj.find("content");
                if (content_it != item_obj.end() && content_it->value().is_array()) {
                    for (auto& part : content_it->value().as_array()) {
                        if (!part.is_object()) continue;
                        auto& p = part.as_object();
                        auto pt_it = p.find("type");
                        if (pt_it == p.end() || !pt_it->value().is_string()) continue;
                        if (std::string(pt_it->value().as_string()) == "output_text") {
                            auto txt_it = p.find("text");
                            if (txt_it != p.end() && txt_it->value().is_string()) {
                                result.content.push_back(TextContent{std::string(txt_it->value().as_string())});
                            }
                        }
                    }
                }
            } else if (type == "function_call") {
                auto cid_it = item_obj.find("call_id");
                auto name_it = item_obj.find("name");
                auto args_it = item_obj.find("arguments");
                if (cid_it != item_obj.end() && cid_it->value().is_string() &&
                    name_it != item_obj.end() && name_it->value().is_string()) {
                    result.content.push_back(ToolCallContent{
                        .tool_call_id = std::string(cid_it->value().as_string()),
                        .tool_name = std::string(name_it->value().as_string()),
                        .input = (args_it != item_obj.end() && args_it->value().is_string())
                            ? std::string(args_it->value().as_string()) : "{}"
                    });
                }
            }
        }
    }

    auto status_it = obj.find("status");
    if (status_it != obj.end() && status_it->value().is_string()) {
        auto status = std::string(status_it->value().as_string());
        if (status == "completed") result.finish_reason = FinishReason::Stop;
        else if (status == "incomplete") result.finish_reason = FinishReason::Length;
        else result.finish_reason = FinishReason::Other;
    }

    auto usage_it = obj.find("usage");
    if (usage_it != obj.end() && usage_it->value().is_object()) {
        auto& u = usage_it->value().as_object();
        if (auto it = u.find("input_tokens"); it != u.end() && it->value().is_int64())
            result.usage.input_tokens.total = static_cast<int>(it->value().as_int64());
        if (auto it = u.find("output_tokens"); it != u.end() && it->value().is_int64())
            result.usage.output_tokens.total = static_cast<int>(it->value().as_int64());
    }

    return result;
}

Task<GenerateResult> OpenAIResponsesModel::do_generate(CallOptions options) {
    auto body = build_request_body(options);
    auto url = provider_.options().base_url + "/responses";
    auto headers = provider_.auth_headers();

    auto response = co_await provider_.http_client().post_json(url, json::value(std::move(body)), headers);
    co_return parse_response(json::parse(response.body));
}

Task<StreamResult> OpenAIResponsesModel::do_stream(CallOptions options) {
    auto body = build_request_body(options, true);
    auto url = provider_.options().base_url + "/responses";
    auto headers = provider_.auth_headers();

    auto response = co_await provider_.http_client().post_streaming(url, json::value(std::move(body)), headers);

    auto stream = [](AsyncGenerator<std::vector<uint8_t>> body_stream) -> AsyncGenerator<StreamPart> {
        std::string buffer;
        std::string current_tool_id;
        bool in_text = false;

        while (auto chunk = co_await body_stream.next()) {
            buffer.append(chunk->begin(), chunk->end());

            // Process SSE events in buffer
            while (true) {
                auto pos = buffer.find("\n\n");
                if (pos == std::string::npos) break;

                auto event_block = buffer.substr(0, pos);
                buffer = buffer.substr(pos + 2);

                std::string event_type, data;
                std::istringstream iss(event_block);
                std::string line;
                while (std::getline(iss, line)) {
                    if (line.starts_with("event: ")) event_type = line.substr(7);
                    else if (line.starts_with("data: ")) data = line.substr(6);
                }

                if (data.empty()) continue;

                boost::system::error_code ec;
                auto val = json::parse(data, ec);
                if (ec) continue;

                if (event_type == "response.output_text.delta") {
                    if (!in_text) {
                        co_yield TextStart{.id = "0"};
                        in_text = true;
                    }
                    if (val.is_object()) {
                        auto d_it = val.as_object().find("delta");
                        if (d_it != val.as_object().end() && d_it->value().is_string()) {
                            co_yield TextDelta{
                                .id = "0",
                                .delta = std::string(d_it->value().as_string())
                            };
                        }
                    }
                } else if (event_type == "response.output_text.done") {
                    if (in_text) {
                        co_yield TextEnd{.id = "0"};
                        in_text = false;
                    }
                } else if (event_type == "response.function_call_arguments.delta") {
                    if (val.is_object()) {
                        auto d_it = val.as_object().find("delta");
                        if (d_it != val.as_object().end() && d_it->value().is_string()) {
                            co_yield ToolInputDelta{
                                .id = current_tool_id,
                                .delta = std::string(d_it->value().as_string())
                            };
                        }
                    }
                } else if (event_type == "response.output_item.added") {
                    if (val.is_object()) {
                        auto& obj2 = val.as_object();
                        auto t_it = obj2.find("type");
                        if (t_it != obj2.end() && t_it->value().is_string() &&
                            std::string(t_it->value().as_string()) == "function_call") {
                            auto cid_it = obj2.find("call_id");
                            auto n_it = obj2.find("name");
                            if (cid_it != obj2.end() && cid_it->value().is_string())
                                current_tool_id = std::string(cid_it->value().as_string());
                            co_yield ToolInputStart{
                                .id = current_tool_id,
                                .tool_name = (n_it != obj2.end() && n_it->value().is_string())
                                    ? std::string(n_it->value().as_string()) : ""
                            };
                        }
                    }
                } else if (event_type == "response.function_call_arguments.done") {
                    co_yield ToolInputEnd{.id = current_tool_id};
                } else if (event_type == "response.completed") {
                    Usage usage;
                    if (val.is_object()) {
                        auto u_it = val.as_object().find("usage");
                        if (u_it != val.as_object().end() && u_it->value().is_object()) {
                            auto& u = u_it->value().as_object();
                            if (auto it = u.find("input_tokens"); it != u.end() && it->value().is_int64())
                                usage.input_tokens.total = static_cast<int>(it->value().as_int64());
                            if (auto it = u.find("output_tokens"); it != u.end() && it->value().is_int64())
                                usage.output_tokens.total = static_cast<int>(it->value().as_int64());
                        }
                    }
                    co_yield FinishPart{.reason = FinishReason::Stop, .usage = usage};
                }
            }
        }
    }(std::move(response.body_stream));

    co_return StreamResult{.stream = std::move(stream)};
}

} // namespace ai::providers::openai
