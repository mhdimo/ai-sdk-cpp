#pragma once

#include <ai/model/language_model.hpp>
#include <ai/prompt/message.hpp>
#include <ai/prompt/content_part.hpp>
#include <boost/json.hpp>
#include <string>
#include <string_view>

namespace ai::workflow {

inline boost::json::object serialize_model_ref(const LanguageModel& model) {
    boost::json::object ref;
    ref["provider"] = std::string(model.provider());
    ref["model_id"] = std::string(model.model_id());
    return ref;
}

inline boost::json::object serialize_call(
    std::string_view model_provider,
    std::string_view model_id,
    const Prompt& prompt,
    const boost::json::object& options = {}
) {
    boost::json::object call;
    call["model"] = boost::json::object{
        {"provider", std::string(model_provider)},
        {"model_id", std::string(model_id)}
    };

    boost::json::array messages;
    for (auto& msg : prompt) {
        std::visit([&](auto& m) {
            using T = std::decay_t<decltype(m)>;
            if constexpr (std::is_same_v<T, SystemMessage>) {
                messages.push_back(boost::json::object{
                    {"role", "system"}, {"content", m.content}
                });
            } else if constexpr (std::is_same_v<T, UserMessage>) {
                std::string text;
                for (auto& part : m.content) {
                    if (auto* t = std::get_if<TextPart>(&part)) text += t->text;
                }
                messages.push_back(boost::json::object{
                    {"role", "user"}, {"content", text}
                });
            } else if constexpr (std::is_same_v<T, AssistantMessage>) {
                std::string text;
                for (auto& part : m.content) {
                    if (auto* t = std::get_if<TextPart>(&part)) text += t->text;
                }
                messages.push_back(boost::json::object{
                    {"role", "assistant"}, {"content", text}
                });
            }
        }, msg);
    }
    call["messages"] = std::move(messages);
    if (!options.empty()) call["options"] = options;
    return call;
}

inline Prompt deserialize_prompt(const boost::json::array& messages) {
    Prompt prompt;
    for (auto& msg : messages) {
        auto& obj = msg.as_object();
        auto role = std::string(obj.at("role").as_string());
        auto content = std::string(obj.at("content").as_string());
        if (role == "system") {
            prompt.push_back(SystemMessage{.content = content});
        } else if (role == "user") {
            UserContent parts;
            parts.push_back(TextPart{.text = content});
            prompt.push_back(UserMessage{.content = std::move(parts)});
        } else if (role == "assistant") {
            AssistantContent parts;
            parts.push_back(TextPart{.text = content});
            prompt.push_back(AssistantMessage{.content = std::move(parts)});
        }
    }
    return prompt;
}

} // namespace ai::workflow
