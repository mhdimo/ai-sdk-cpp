#include <ai/providers/bedrock/bedrock.hpp>
#include <ai/model/language_model.hpp>
#include <ai/model/call_options.hpp>
#include <ai/model/generate_result.hpp>
#include <ai/model/stream_result.hpp>
#include <ai/stream/stream_part.hpp>
#include <ai/stream/async_generator.hpp>
#include <ai/http/client.hpp>
#include <ai/util/json.hpp>
#include <boost/json.hpp>
#include <stdexcept>
#include <cstdint>

namespace ai::providers::bedrock {

namespace {

namespace json = boost::json;

FinishReason map_stop_reason(std::string_view reason) {
    if (reason == "end_turn" || reason == "stop") return FinishReason::Stop;
    if (reason == "tool_use") return FinishReason::ToolCalls;
    if (reason == "max_tokens") return FinishReason::Length;
    if (reason == "stop_sequence") return FinishReason::Stop;
    if (reason == "content_filtered") return FinishReason::ContentFilter;
    return FinishReason::Other;
}

json::array convert_content_parts(const UserContent& content) {
    json::array parts;
    for (auto& part : content) {
        if (auto* text = std::get_if<TextPart>(&part)) {
            parts.push_back(json::object{{"text", text->text}});
        }
    }
    return parts;
}

json::array convert_assistant_content(const AssistantContent& content) {
    json::array parts;
    for (auto& part : content) {
        if (auto* text = std::get_if<TextPart>(&part)) {
            parts.push_back(json::object{{"text", text->text}});
        } else if (auto* tc = std::get_if<ToolCallPart>(&part)) {
            auto input_str = json::serialize(tc->input);
            json::value input_val = json::parse(input_str);
            parts.push_back(json::object{
                {"toolUse", json::object{
                    {"toolUseId", tc->tool_call_id},
                    {"name", tc->tool_name},
                    {"input", input_val}
                }}
            });
        }
    }
    return parts;
}

json::array convert_tool_results(const std::vector<ToolResultPart>& results) {
    json::array parts;
    for (auto& r : results) {
        json::object tool_result;
        tool_result["toolUseId"] = r.tool_call_id;

        json::array content;
        std::string status = "success";

        std::visit([&](auto& output) {
            using T = std::decay_t<decltype(output)>;
            if constexpr (std::is_same_v<T, TextOutput>) {
                content.push_back(json::object{{"text", output.value}});
            } else if constexpr (std::is_same_v<T, JsonOutput>) {
                if (output.value.is_object()) {
                    content.push_back(json::object{{"json", output.value}});
                } else {
                    content.push_back(json::object{{"text", json::serialize(output.value)}});
                }
            } else if constexpr (std::is_same_v<T, ErrorTextOutput>) {
                content.push_back(json::object{{"text", output.value}});
                status = "error";
            } else if constexpr (std::is_same_v<T, ErrorJsonOutput>) {
                if (output.value.is_string()) {
                    content.push_back(json::object{{"text", std::string(output.value.as_string())}});
                } else {
                    content.push_back(json::object{{"text", json::serialize(output.value)}});
                }
                status = "error";
            } else if constexpr (std::is_same_v<T, ExecutionDenied>) {
                content.push_back(json::object{{"text", output.reason.value_or("Tool execution denied")}});
                status = "error";
            } else if constexpr (std::is_same_v<T, ContentOutput>) {
                for (auto& part : output.value) {
                    std::visit([&](auto& p) {
                        using P = std::decay_t<decltype(p)>;
                        if constexpr (std::is_same_v<P, TextPart>) {
                            content.push_back(json::object{{"text", p.text}});
                        }
                    }, part.part);
                }
            }
        }, r.output);

        tool_result["content"] = std::move(content);
        if (status != "success") {
            tool_result["status"] = status;
        }
        parts.push_back(json::object{{"toolResult", std::move(tool_result)}});
    }
    return parts;
}

json::object build_request_body(const CallOptions& options) {
    json::object body;

    // System messages
    json::array system_parts;
    json::array messages;

    for (auto& msg : options.prompt) {
        if (auto* sys = std::get_if<SystemMessage>(&msg)) {
            system_parts.push_back(json::object{{"text", sys->content}});
        } else if (auto* user = std::get_if<UserMessage>(&msg)) {
            auto parts = convert_content_parts(user->content);
            if (!parts.empty()) {
                messages.push_back(json::object{
                    {"role", "user"},
                    {"content", std::move(parts)}
                });
            }
        } else if (auto* asst = std::get_if<AssistantMessage>(&msg)) {
            auto parts = convert_assistant_content(asst->content);
            if (!parts.empty()) {
                messages.push_back(json::object{
                    {"role", "assistant"},
                    {"content", std::move(parts)}
                });
            }
        } else if (auto* tool = std::get_if<ToolMessage>(&msg)) {
            auto parts = convert_tool_results(tool->content);
            if (!parts.empty()) {
                messages.push_back(json::object{
                    {"role", "user"},
                    {"content", std::move(parts)}
                });
            }
        }
    }

    if (!system_parts.empty()) body["system"] = std::move(system_parts);
    body["messages"] = std::move(messages);

    // Inference config
    json::object inference_config;
    if (options.max_output_tokens)
        inference_config["maxTokens"] = *options.max_output_tokens;
    if (options.temperature)
        inference_config["temperature"] = *options.temperature;
    if (options.top_p)
        inference_config["topP"] = *options.top_p;
    if (options.stop_sequences && !options.stop_sequences->empty()) {
        json::array stops;
        for (auto& s : *options.stop_sequences) stops.push_back(json::value(s));
        inference_config["stopSequences"] = std::move(stops);
    }
    if (!inference_config.empty()) body["inferenceConfig"] = std::move(inference_config);

    // Tools
    if (!options.tools.empty()) {
        json::array tool_specs;
        for (auto& t : options.tools) {
            json::object spec;
            spec["name"] = t.name;
            if (t.description) spec["description"] = *t.description;
            spec["inputSchema"] = json::object{{"json", json::parse(t.input_schema.to_string())}};
            tool_specs.push_back(json::object{{"toolSpec", std::move(spec)}});
        }

        json::object tool_config;
        tool_config["tools"] = std::move(tool_specs);

        if (options.tool_choice) {
            std::visit([&](auto& choice) {
                using T = std::decay_t<decltype(choice)>;
                if constexpr (std::is_same_v<T, ToolChoiceAuto>) {
                    tool_config["toolChoice"] = json::object{{"auto", json::object{}}};
                } else if constexpr (std::is_same_v<T, ToolChoiceRequired>) {
                    tool_config["toolChoice"] = json::object{{"any", json::object{}}};
                } else if constexpr (std::is_same_v<T, ToolChoiceNone>) {
                    // Bedrock doesn't have a "none" - omit toolConfig entirely
                } else if constexpr (std::is_same_v<T, ToolChoiceSpecific>) {
                    tool_config["toolChoice"] = json::object{
                        {"tool", json::object{{"name", json::value(choice.tool_name)}}}
                    };
                }
            }, *options.tool_choice);
        }

        body["toolConfig"] = std::move(tool_config);
    }

    return body;
}

GenerateResult parse_response(const json::value& response) {
    GenerateResult result;
    if (!response.is_object()) return result;
    auto& resp_obj = response.as_object();

    auto output_it = resp_obj.find("output");
    if (output_it == resp_obj.end() || !output_it->value().is_object()) return result;
    auto& output = output_it->value().as_object();

    auto msg_it = output.find("message");
    if (msg_it == output.end() || !msg_it->value().is_object()) return result;
    auto& message = msg_it->value().as_object();

    auto content_it = message.find("content");
    if (content_it == message.end() || !content_it->value().is_array()) return result;

    for (auto& part : content_it->value().as_array()) {
        if (!part.is_object()) continue;
        auto& obj = part.as_object();
        if (auto t = obj.find("text"); t != obj.end() && t->value().is_string()) {
            result.content.push_back(TextContent{std::string(t->value().as_string())});
        } else if (auto tu_it = obj.find("toolUse"); tu_it != obj.end() && tu_it->value().is_object()) {
            auto& tu = tu_it->value().as_object();
            auto id_it = tu.find("toolUseId");
            auto name_it = tu.find("name");
            auto input_it = tu.find("input");
            if (id_it != tu.end() && id_it->value().is_string() &&
                name_it != tu.end() && name_it->value().is_string()) {
                result.content.push_back(ToolCallContent{
                    .tool_call_id = std::string(id_it->value().as_string()),
                    .tool_name = std::string(name_it->value().as_string()),
                    .input = input_it != tu.end() ? json::serialize(input_it->value()) : "{}"
                });
            }
        } else if (auto rc_it = obj.find("reasoningContent"); rc_it != obj.end() && rc_it->value().is_object()) {
            auto& rc = rc_it->value().as_object();
            auto rt_it = rc.find("reasoningText");
            if (rt_it != rc.end() && rt_it->value().is_object()) {
                auto& rt = rt_it->value().as_object();
                auto txt_it = rt.find("text");
                if (txt_it != rt.end() && txt_it->value().is_string()) {
                    result.content.push_back(ReasoningContent{
                        std::string(txt_it->value().as_string())
                    });
                }
            }
        }
    }

    auto sr_it = resp_obj.find("stopReason");
    if (sr_it != resp_obj.end() && sr_it->value().is_string()) {
        result.finish_reason = map_stop_reason(sr_it->value().as_string());
    }

    auto usage_it = resp_obj.find("usage");
    if (usage_it != resp_obj.end() && usage_it->value().is_object()) {
        auto& u = usage_it->value().as_object();
        if (auto it = u.find("inputTokens"); it != u.end() && it->value().is_int64())
            result.usage.input_tokens.total = static_cast<int>(it->value().as_int64());
        if (auto it = u.find("outputTokens"); it != u.end() && it->value().is_int64())
            result.usage.output_tokens.total = static_cast<int>(it->value().as_int64());
    }

    return result;
}

// Binary event stream decoder for Bedrock streaming
struct EventStreamEvent {
    std::string event_type;
    std::string content_type;
    std::string payload;
};

std::optional<EventStreamEvent> decode_event(const std::vector<uint8_t>& data, size_t& offset) {
    if (offset + 12 > data.size()) return std::nullopt;

    uint32_t total_length = (uint32_t(data[offset]) << 24) | (uint32_t(data[offset+1]) << 16) |
                            (uint32_t(data[offset+2]) << 8) | uint32_t(data[offset+3]);
    uint32_t headers_length = (uint32_t(data[offset+4]) << 24) | (uint32_t(data[offset+5]) << 16) |
                              (uint32_t(data[offset+6]) << 8) | uint32_t(data[offset+7]);

    if (offset + total_length > data.size()) return std::nullopt;

    // Skip prelude CRC (4 bytes after total_length + headers_length)
    size_t headers_start = offset + 12;
    size_t headers_end = headers_start + headers_length;
    size_t payload_start = headers_end;
    size_t payload_end = offset + total_length - 4; // minus message CRC

    EventStreamEvent event;

    // Parse headers
    size_t hpos = headers_start;
    while (hpos < headers_end) {
        uint8_t name_len = data[hpos++];
        std::string name(data.begin() + hpos, data.begin() + hpos + name_len);
        hpos += name_len;
        uint8_t header_type = data[hpos++];
        if (header_type == 7) { // String type
            uint16_t val_len = (uint16_t(data[hpos]) << 8) | uint16_t(data[hpos+1]);
            hpos += 2;
            std::string value(data.begin() + hpos, data.begin() + hpos + val_len);
            hpos += val_len;

            if (name == ":event-type") event.event_type = value;
            else if (name == ":content-type") event.content_type = value;
        } else {
            break; // Unknown header type, skip rest
        }
    }

    if (payload_start < payload_end) {
        event.payload = std::string(data.begin() + payload_start, data.begin() + payload_end);
    }

    offset += total_length;
    return event;
}

std::vector<StreamPart> parse_stream_event(const EventStreamEvent& event) {
    std::vector<StreamPart> parts;
    if (event.payload.empty()) return parts;

    boost::system::error_code ec;
    auto val = json::parse(event.payload, ec);
    if (ec || !val.is_object()) return parts;
    auto& obj = val.as_object();

    auto cbs_it = obj.find("contentBlockStart");
    if (cbs_it != obj.end() && cbs_it->value().is_object()) {
        auto& block = cbs_it->value().as_object();
        auto idx_it = block.find("contentBlockIndex");
        std::string idx = (idx_it != block.end() && idx_it->value().is_int64())
            ? std::to_string(idx_it->value().as_int64()) : "0";
        auto start_it = block.find("start");
        if (start_it != block.end() && start_it->value().is_object()) {
            auto& start = start_it->value().as_object();
            auto tu_it = start.find("toolUse");
            if (tu_it != start.end() && tu_it->value().is_object()) {
                auto& tu = tu_it->value().as_object();
                auto id_it = tu.find("toolUseId");
                auto name_it = tu.find("name");
                parts.push_back(ToolInputStart{
                    .id = (id_it != tu.end() && id_it->value().is_string())
                        ? std::string(id_it->value().as_string()) : idx,
                    .tool_name = (name_it != tu.end() && name_it->value().is_string())
                        ? std::string(name_it->value().as_string()) : "",
                });
            } else {
                parts.push_back(TextStart{.id = idx});
            }
        }
        return parts;
    }

    auto cbd_it = obj.find("contentBlockDelta");
    if (cbd_it != obj.end() && cbd_it->value().is_object()) {
        auto& block = cbd_it->value().as_object();
        auto idx_it = block.find("contentBlockIndex");
        std::string idx = (idx_it != block.end() && idx_it->value().is_int64())
            ? std::to_string(idx_it->value().as_int64()) : "0";
        auto delta_it = block.find("delta");
        if (delta_it != block.end() && delta_it->value().is_object()) {
            auto& delta = delta_it->value().as_object();
            if (auto t = delta.find("text"); t != delta.end() && t->value().is_string()) {
                parts.push_back(TextDelta{
                    .id = idx,
                    .delta = std::string(t->value().as_string())
                });
            } else if (auto tu_it2 = delta.find("toolUse"); tu_it2 != delta.end() && tu_it2->value().is_object()) {
                auto& tu = tu_it2->value().as_object();
                auto inp_it = tu.find("input");
                if (inp_it != tu.end() && inp_it->value().is_string()) {
                    parts.push_back(ToolInputDelta{
                        .id = idx,
                        .delta = std::string(inp_it->value().as_string())
                    });
                }
            } else if (auto rc_it = delta.find("reasoningContent"); rc_it != delta.end() && rc_it->value().is_object()) {
                auto& rc = rc_it->value().as_object();
                auto txt_it = rc.find("text");
                if (txt_it != rc.end() && txt_it->value().is_string()) {
                    parts.push_back(ReasoningDelta{
                        .id = idx,
                        .delta = std::string(txt_it->value().as_string())
                    });
                }
            }
        }
        return parts;
    }

    auto cbstop_it = obj.find("contentBlockStop");
    if (cbstop_it != obj.end() && cbstop_it->value().is_object()) {
        auto& block = cbstop_it->value().as_object();
        auto idx_it = block.find("contentBlockIndex");
        std::string idx = (idx_it != block.end() && idx_it->value().is_int64())
            ? std::to_string(idx_it->value().as_int64()) : "0";
        parts.push_back(TextEnd{.id = idx});
        return parts;
    }

    auto ms_it = obj.find("messageStop");
    if (ms_it != obj.end() && ms_it->value().is_object()) {
        auto& stop = ms_it->value().as_object();
        FinishReason reason = FinishReason::Stop;
        auto sr_it = stop.find("stopReason");
        if (sr_it != stop.end() && sr_it->value().is_string()) {
            reason = map_stop_reason(sr_it->value().as_string());
        }
        parts.push_back(FinishPart{.reason = reason, .usage = {}});
        return parts;
    }

    auto meta_it = obj.find("metadata");
    if (meta_it != obj.end() && meta_it->value().is_object()) {
        auto& meta = meta_it->value().as_object();
        auto u_it = meta.find("usage");
        if (u_it != meta.end() && u_it->value().is_object()) {
            auto& u = u_it->value().as_object();
            Usage usage;
            if (auto it = u.find("inputTokens"); it != u.end() && it->value().is_int64())
                usage.input_tokens.total = static_cast<int>(it->value().as_int64());
            if (auto it = u.find("outputTokens"); it != u.end() && it->value().is_int64())
                usage.output_tokens.total = static_cast<int>(it->value().as_int64());
            parts.push_back(FinishPart{.reason = FinishReason::Stop, .usage = usage});
        }
    }

    return parts;
}

} // namespace

class BedrockLanguageModel : public LanguageModel {
public:
    BedrockLanguageModel(std::string model_id, BedrockProvider& provider)
        : model_id_(std::move(model_id)), provider_(provider) {}

    std::string_view provider() const override { return "amazon-bedrock"; }
    std::string_view model_id() const override { return model_id_; }

    Task<GenerateResult> do_generate(CallOptions options) override {
        json::object body = build_request_body(options);
        auto body_str = json::serialize(body);

        auto url = provider_.runtime_base_url() + "/model/" + model_id_ + "/converse";
        auto headers = provider_.auth_headers(url, body_str);

        auto response = co_await provider_.http_client().post_json(url, json::value(std::move(body)), headers);
        auto result = parse_response(json::parse(response.body));
        co_return result;
    }

    Task<StreamResult> do_stream(CallOptions options) override {
        json::object body = build_request_body(options);
        auto body_str = json::serialize(body);

        auto url = provider_.runtime_base_url() + "/model/" + model_id_ + "/converse-stream";
        auto headers = provider_.auth_headers(url, body_str);

        auto response = co_await provider_.http_client().post_streaming(url, json::value(std::move(body)), headers);

        auto stream = [](AsyncGenerator<std::vector<uint8_t>> body_stream) -> AsyncGenerator<StreamPart> {
            std::vector<uint8_t> buffer;

            while (auto chunk = co_await body_stream.next()) {
                buffer.insert(buffer.end(), chunk->begin(), chunk->end());

                size_t offset = 0;
                while (auto event = decode_event(buffer, offset)) {
                    if (!event->payload.empty()) {
                        auto parts = parse_stream_event(*event);
                        for (auto& part : parts) {
                            co_yield std::move(part);
                        }
                    }
                }

                if (offset > 0) {
                    buffer.erase(buffer.begin(), buffer.begin() + offset);
                }
            }
        }(std::move(response.body_stream));

        co_return StreamResult{.stream = std::move(stream)};
    }

private:
    std::string model_id_;
    BedrockProvider& provider_;
};

LanguageModelPtr create_bedrock_model(std::string model_id, BedrockProvider& provider) {
    return std::make_shared<BedrockLanguageModel>(std::move(model_id), provider);
}

} // namespace ai::providers::bedrock
