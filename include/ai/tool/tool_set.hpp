#pragma once

#include <ai/tool/tool.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <optional>

namespace ai {

class ToolSet {
public:
    ToolSet() = default;

    void add(ToolDefinition tool) {
        auto name = tool.name;
        tools_.emplace(std::move(name), std::move(tool));
    }

    const ToolDefinition* find(std::string_view name) const {
        auto it = tools_.find(std::string(name));
        if (it == tools_.end()) return nullptr;
        return &it->second;
    }

    std::vector<const ToolDefinition*> all() const {
        std::vector<const ToolDefinition*> result;
        result.reserve(tools_.size());
        for (auto& [_, tool] : tools_) {
            result.push_back(&tool);
        }
        return result;
    }

    size_t size() const { return tools_.size(); }
    bool empty() const { return tools_.empty(); }

    auto begin() const { return tools_.begin(); }
    auto end() const { return tools_.end(); }

private:
    std::unordered_map<std::string, ToolDefinition> tools_;
};

} // namespace ai
