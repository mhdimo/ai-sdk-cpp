#pragma once

#include <string>
#include <boost/json.hpp>

namespace ai {

struct ToolCall {
    std::string tool_call_id;
    std::string tool_name;
    std::string input; // raw JSON string
    bool provider_executed = false;
};

struct ToolCallResult {
    std::string tool_call_id;
    std::string tool_name;
    boost::json::value input;
    boost::json::value output;
    bool is_error = false;
};

} // namespace ai
