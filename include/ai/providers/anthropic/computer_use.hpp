#pragma once

#include <boost/json.hpp>

namespace ai::providers::anthropic {

struct ComputerUseToolConfig {
    int display_width = 1024;
    int display_height = 768;
    int display_number = 1;
};

struct TextEditorToolConfig {
    // no config needed
};

struct BashToolConfig {
    // no config needed
};

// Returns the provider_options object to add these tools
inline boost::json::object computer_use_tools(
    ComputerUseToolConfig computer = {},
    bool include_text_editor = true,
    bool include_bash = true
) {
    boost::json::array tools;

    boost::json::object computer_tool;
    computer_tool["type"] = "computer_20241022";
    computer_tool["name"] = "computer";
    computer_tool["display_width_px"] = computer.display_width;
    computer_tool["display_height_px"] = computer.display_height;
    computer_tool["display_number"] = computer.display_number;
    tools.push_back(std::move(computer_tool));

    if (include_text_editor) {
        tools.push_back(boost::json::object{
            {"type", "text_editor_20241022"},
            {"name", "str_replace_editor"}
        });
    }

    if (include_bash) {
        tools.push_back(boost::json::object{
            {"type", "bash_20241022"},
            {"name", "bash"}
        });
    }

    boost::json::object result;
    result["anthropic"] = boost::json::object{{"builtinTools", std::move(tools)}};
    return result;
}

} // namespace ai::providers::anthropic
