#pragma once

#include <boost/json.hpp>
#include <string>
#include <string_view>
#include <optional>
#include <stdexcept>

namespace ai::json {

class ParseError : public std::runtime_error {
public:
    ParseError(std::string_view text, std::string_view reason)
        : std::runtime_error(std::string("JSON parse error: ") + std::string(reason))
        , text_(text) {}

    std::string_view text() const noexcept { return text_; }

private:
    std::string text_;
};

inline boost::json::value parse(std::string_view text) {
    boost::system::error_code ec;
    auto val = boost::json::parse(text, ec);
    if (ec) {
        throw ParseError(text, ec.message());
    }
    return val;
}

inline std::optional<boost::json::value> safe_parse(std::string_view text) noexcept {
    boost::system::error_code ec;
    auto val = boost::json::parse(text, ec);
    if (ec) return std::nullopt;
    return val;
}

inline std::string serialize(const boost::json::value& val) {
    return boost::json::serialize(val);
}

inline std::optional<std::string> get_string(const boost::json::value& v, std::string_view key) {
    if (!v.is_object()) return std::nullopt;
    auto& obj = v.as_object();
    auto it = obj.find(key);
    if (it == obj.end() || !it->value().is_string()) return std::nullopt;
    return std::string(it->value().as_string());
}

inline std::optional<int64_t> get_int(const boost::json::value& v, std::string_view key) {
    if (!v.is_object()) return std::nullopt;
    auto& obj = v.as_object();
    auto it = obj.find(key);
    if (it == obj.end() || !it->value().is_int64()) return std::nullopt;
    return it->value().as_int64();
}

inline std::optional<double> get_double(const boost::json::value& v, std::string_view key) {
    if (!v.is_object()) return std::nullopt;
    auto& obj = v.as_object();
    auto it = obj.find(key);
    if (it == obj.end()) return std::nullopt;
    if (it->value().is_double()) return it->value().as_double();
    if (it->value().is_int64()) return static_cast<double>(it->value().as_int64());
    return std::nullopt;
}

inline bool get_bool(const boost::json::value& v, std::string_view key, bool default_val = false) {
    if (!v.is_object()) return default_val;
    auto& obj = v.as_object();
    auto it = obj.find(key);
    if (it == obj.end() || !it->value().is_bool()) return default_val;
    return it->value().as_bool();
}

} // namespace ai::json
