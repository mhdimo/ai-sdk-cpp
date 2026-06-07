#pragma once

#include <string>
#include <variant>

namespace ai {

struct ToolChoiceAuto {};
struct ToolChoiceNone {};
struct ToolChoiceRequired {};
struct ToolChoiceSpecific {
    std::string tool_name;
};

using ToolChoice = std::variant<ToolChoiceAuto, ToolChoiceNone, ToolChoiceRequired, ToolChoiceSpecific>;

} // namespace ai
