#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ai::util {

namespace detail {

constexpr char base64_chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

} // namespace detail

inline std::string base64_encode(const std::vector<uint8_t>& data) {
    std::string result;
    result.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    while (i + 2 < data.size()) {
        uint32_t triple = (static_cast<uint32_t>(data[i]) << 16)
                        | (static_cast<uint32_t>(data[i + 1]) << 8)
                        | static_cast<uint32_t>(data[i + 2]);
        result += detail::base64_chars[(triple >> 18) & 0x3F];
        result += detail::base64_chars[(triple >> 12) & 0x3F];
        result += detail::base64_chars[(triple >> 6) & 0x3F];
        result += detail::base64_chars[triple & 0x3F];
        i += 3;
    }

    if (i < data.size()) {
        uint32_t triple = static_cast<uint32_t>(data[i]) << 16;
        if (i + 1 < data.size()) {
            triple |= static_cast<uint32_t>(data[i + 1]) << 8;
        }
        result += detail::base64_chars[(triple >> 18) & 0x3F];
        result += detail::base64_chars[(triple >> 12) & 0x3F];
        if (i + 1 < data.size()) {
            result += detail::base64_chars[(triple >> 6) & 0x3F];
        } else {
            result += '=';
        }
        result += '=';
    }

    return result;
}

inline std::string base64_encode(const uint8_t* data, size_t len) {
    return base64_encode(std::vector<uint8_t>(data, data + len));
}

} // namespace ai::util
