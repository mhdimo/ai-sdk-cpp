#include <catch2/catch_test_macros.hpp>

#include <ai/util/base64.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// base64_encode ships to users through three providers — Anthropic, OpenAI and
// Google all call it to inline image and file bytes as `data:<type>;base64,...`
// — and until this file existed nothing executed it. A padding bug here does not
// throw; it produces a payload the API rejects, or worse, accepts as corrupt.
//
// The vectors below are RFC 4648 §10, which is the only authority worth testing
// against: hand-written expectations would just restate the implementation.

namespace {

std::vector<uint8_t> bytes_of(std::string_view s) {
    return std::vector<uint8_t>(s.begin(), s.end());
}

/// '=' count at the end of `s` (0 when there is none).
size_t trailing_padding(const std::string& s) {
    size_t n = 0;
    while (n < s.size() && s[s.size() - 1 - n] == '=') {
        ++n;
    }
    return n;
}

/// True if a '=' is followed by anything other than more '=' — i.e. padding
/// leaked into the middle of the output.
bool has_interior_padding(const std::string& s) {
    auto first = s.find('=');
    return first != std::string::npos && s.find_first_not_of('=', first) != std::string::npos;
}

/// Encoded length of `n` bytes: four characters per three, rounded up.
size_t encoded_length(size_t n) {
    return 4 * ((n + 2) / 3);
}

} // namespace

TEST_CASE("base64_encode matches the RFC 4648 vectors", "[base64]") {
    REQUIRE(ai::util::base64_encode(bytes_of("")) == "");
    REQUIRE(ai::util::base64_encode(bytes_of("f")) == "Zg==");
    REQUIRE(ai::util::base64_encode(bytes_of("fo")) == "Zm8=");
    REQUIRE(ai::util::base64_encode(bytes_of("foo")) == "Zm9v");
    REQUIRE(ai::util::base64_encode(bytes_of("foob")) == "Zm9vYg==");
    REQUIRE(ai::util::base64_encode(bytes_of("fooba")) == "Zm9vYmE=");
    REQUIRE(ai::util::base64_encode(bytes_of("foobar")) == "Zm9vYmFy");
}

TEST_CASE("base64_encode uses the right length and padding for every remainder", "[base64]") {
    // Sweeping lengths catches an off-by-one in the `i + 2 < size` loop bound,
    // which would mis-place the boundary between full and partial quanta.
    for (size_t n = 0; n < 10; ++n) {
        std::vector<uint8_t> data(n, 'A');
        auto encoded = ai::util::base64_encode(data);

        INFO("input length " << n);
        REQUIRE(encoded.size() == encoded_length(n));
        REQUIRE(encoded.size() % 4 == 0);
        REQUIRE_FALSE(has_interior_padding(encoded));
        REQUIRE(trailing_padding(encoded) == (3 - n % 3) % 3);
    }
}

TEST_CASE("base64_encode handles the byte values that sit on bit boundaries", "[base64]") {
    // 0x00 and 0xFF are where a shift-by-6 goes wrong: all-zero spans the whole
    // alphabet, all-ones exercises the `+` and `/` end of the table.
    REQUIRE(ai::util::base64_encode(std::vector<uint8_t>{0x00}) == "AA==");
    REQUIRE(ai::util::base64_encode(std::vector<uint8_t>{0xFF}) == "/w==");
    REQUIRE(ai::util::base64_encode(std::vector<uint8_t>{0xFF, 0xFF, 0xFF}) == "////");
    REQUIRE(ai::util::base64_encode(std::vector<uint8_t>{0x00, 0x00, 0x00}) == "AAAA");
    REQUIRE(ai::util::base64_encode(std::vector<uint8_t>{0xFB, 0xFF}) == "+/8=");
}

TEST_CASE("base64_encode emits only alphabet characters plus padding", "[base64]") {
    // Every possible byte, in every position of the quantum, so no shift can
    // index outside base64_chars.
    const std::string alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";

    std::vector<uint8_t> all_bytes(256);
    for (int b = 0; b < 256; ++b) {
        all_bytes[static_cast<size_t>(b)] = static_cast<uint8_t>(b);
    }

    for (size_t len : {1U, 2U, 3U, 4U, 255U, 256U}) {
        std::vector<uint8_t> data(all_bytes.begin(), all_bytes.begin() + len);
        auto encoded = ai::util::base64_encode(data);

        INFO("length " << len);
        REQUIRE(encoded.find_first_not_of(alphabet) == std::string::npos);
        REQUIRE(encoded.size() == encoded_length(len));
    }
}

TEST_CASE("base64_encode pointer overload agrees with the vector overload", "[base64]") {
    auto data = bytes_of("the quick brown fox");
    REQUIRE(ai::util::base64_encode(data.data(), data.size()) ==
            ai::util::base64_encode(data));
    // A zero-length pointer range must not read through the pointer.
    REQUIRE(ai::util::base64_encode(data.data(), 0) == "");
}
