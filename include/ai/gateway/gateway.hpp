#pragma once

#include <ai/http/client.hpp>
#include <string>
#include <string_view>
#include <unordered_map>

namespace ai::gateway {

struct GatewayConfig {
    std::string base_url;
    std::unordered_map<std::string, std::string> headers;
};

inline std::string rewrite_url(
    const GatewayConfig& gateway,
    std::string_view provider_name,
    std::string_view original_path
) {
    return gateway.base_url + "/" + std::string(provider_name) + std::string(original_path);
}

inline http::Headers apply_gateway_headers(
    const GatewayConfig& gateway,
    http::Headers existing
) {
    for (auto& [k, v] : gateway.headers) {
        existing[k] = v;
    }
    return existing;
}

} // namespace ai::gateway
