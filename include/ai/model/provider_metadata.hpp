#pragma once

#include <boost/json.hpp>
#include <string>
#include <string_view>
#include <optional>

namespace ai::provider_metadata {

inline std::optional<std::string> get_string(
    const boost::json::object& metadata,
    std::string_view provider,
    std::string_view key
) {
    auto it = metadata.find(provider);
    if (it == metadata.end()) return std::nullopt;
    if (!it->value().is_object()) return std::nullopt;
    auto& obj = it->value().as_object();
    auto kit = obj.find(key);
    if (kit == obj.end()) return std::nullopt;
    if (!kit->value().is_string()) return std::nullopt;
    return std::string(kit->value().as_string());
}

inline std::optional<int> get_int(
    const boost::json::object& metadata,
    std::string_view provider,
    std::string_view key
) {
    auto it = metadata.find(provider);
    if (it == metadata.end()) return std::nullopt;
    if (!it->value().is_object()) return std::nullopt;
    auto& obj = it->value().as_object();
    auto kit = obj.find(key);
    if (kit == obj.end()) return std::nullopt;
    if (!kit->value().is_int64()) return std::nullopt;
    return (int)kit->value().as_int64();
}

inline void merge(boost::json::object& target, const boost::json::object& source) {
    for (auto& [k, v] : source) {
        target[k] = v;
    }
}

} // namespace ai::provider_metadata
