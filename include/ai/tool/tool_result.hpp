#pragma once

#include <string>
#include <boost/json.hpp>

namespace ai {

struct ToolExecutionResult {
    boost::json::value output;
    bool is_error = false;
};

} // namespace ai
