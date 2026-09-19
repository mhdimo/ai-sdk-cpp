#include <catch2/catch_test_macros.hpp>

#include <ai/stream/data_stream.hpp>
#include <ai/stream/stream_part.hpp>

#include <boost/json.hpp>

#include <optional>
#include <string>
#include <variant>

// The data stream protocol is a wire format, so the exact bytes matter: a
// client written against the documented codes breaks if a prefix or a key name
// drifts, and nothing else in the suite would notice. The encoder and decoder
// below are each other's only consumer in-tree, which means a symmetric mistake
// (both sides agreeing on the wrong thing) would round-trip cleanly and still be
// wrong. The literal expectations are therefore spelled out rather than derived
// from the encoder.

namespace {

using ai::stream::DataStreamDecoder;
using ai::stream::DataStreamEncoder;

std::string encoded(const ai::StreamPart& part) {
    DataStreamEncoder encoder;
    return encoder.encode(part);
}

/// Decode, requiring that the line was understood at all.
ai::StreamPart decoded(const std::string& line) {
    DataStreamDecoder decoder;
    auto part = decoder.decode_line(line);
    REQUIRE(part.has_value());
    return *part;
}

/// What the encoder emits for a part it has no wire representation for. The
/// caller cannot distinguish this from an empty payload, which is worth knowing.
bool has_no_encoding(const ai::StreamPart& part) {
    return encoded(part).empty();
}

ai::Usage usage_of(int input, int output) {
    ai::Usage usage;
    usage.input_tokens.total = input;
    usage.output_tokens.total = output;
    return usage;
}

} // namespace

TEST_CASE("the encoder emits the documented code prefix for each part", "[data_stream]") {
    // The prefix is the whole contract with the client.
    CHECK(encoded(ai::TextDelta{.id = "t", .delta = "hi"}) == "0:\"hi\"\n");
    CHECK(encoded(ai::ToolInputEnd{.id = "call-1"}) == "3:{\"toolCallId\":\"call-1\"}\n");
    CHECK(encoded(ai::ErrorPart{.message = "boom"}) == "e:\"boom\"\n");

    CHECK(encoded(ai::ToolInputStart{.id = "call-1", .tool_name = "search"})
          == "1:{\"toolCallId\":\"call-1\",\"toolName\":\"search\"}\n");
    CHECK(encoded(ai::ToolInputDelta{.id = "call-1", .delta = "{\"q\":"})
          == "2:{\"toolCallId\":\"call-1\",\"argsTextDelta\":\"{\\\"q\\\":\"}\n");
}

TEST_CASE("text deltas round-trip through the encoder and decoder", "[data_stream]") {
    // Including the characters that have to be escaped for the framing to hold:
    // a literal newline inside a delta must not be mistaken for a line break.
    for (const std::string text : {std::string("hello"), std::string(""),
                                   std::string("line\nbreak"), std::string("\"quoted\""),
                                   std::string("uni\u00e9code"), std::string("0:1:2:")}) {
        INFO("delta: " << text);
        auto part = decoded(encoded(ai::TextDelta{.id = "t", .delta = text}));
        REQUIRE(std::holds_alternative<ai::TextDelta>(part));
        CHECK(std::get<ai::TextDelta>(part).delta == text);
    }
}

TEST_CASE("tool call parts round-trip through the encoder and decoder", "[data_stream]") {
    SECTION("start carries the id and the tool name") {
        auto part = decoded(encoded(ai::ToolInputStart{.id = "c1", .tool_name = "read_file"}));
        REQUIRE(std::holds_alternative<ai::ToolInputStart>(part));
        CHECK(std::get<ai::ToolInputStart>(part).id == "c1");
        CHECK(std::get<ai::ToolInputStart>(part).tool_name == "read_file");
    }

    SECTION("delta carries the raw argument fragment") {
        auto part = decoded(encoded(ai::ToolInputDelta{.id = "c1", .delta = "{\"path\":\"a\"}"}));
        REQUIRE(std::holds_alternative<ai::ToolInputDelta>(part));
        CHECK(std::get<ai::ToolInputDelta>(part).id == "c1");
        CHECK(std::get<ai::ToolInputDelta>(part).delta == "{\"path\":\"a\"}");
    }

    SECTION("end carries only the id") {
        auto part = decoded(encoded(ai::ToolInputEnd{.id = "c1"}));
        REQUIRE(std::holds_alternative<ai::ToolInputEnd>(part));
        CHECK(std::get<ai::ToolInputEnd>(part).id == "c1");
    }
}

TEST_CASE("finish parts round-trip for every reason the protocol can express", "[data_stream]") {
    // Only three of the six FinishReason values have a wire spelling. The other
    // three collapse to "other" and come back as Stop, so the round trip is
    // lossy — pinned here so the collapse is a known property, not a surprise.
    struct Case {
        ai::FinishReason sent;
        const char* wire;
        ai::FinishReason received;
    };

    for (const Case& c : {
             Case{ai::FinishReason::Stop, "stop", ai::FinishReason::Stop},
             Case{ai::FinishReason::Length, "length", ai::FinishReason::Length},
             Case{ai::FinishReason::ToolCalls, "tool-calls", ai::FinishReason::ToolCalls},
             Case{ai::FinishReason::ContentFilter, "other", ai::FinishReason::Stop},
             Case{ai::FinishReason::Error, "other", ai::FinishReason::Stop},
             Case{ai::FinishReason::Other, "other", ai::FinishReason::Stop},
         }) {
        INFO("wire spelling: " << c.wire);
        auto line = encoded(ai::FinishPart{.reason = c.sent, .usage = usage_of(1, 2)});
        CHECK(line.find("\"finishReason\":\"" + std::string(c.wire) + "\"") != std::string::npos);

        auto part = decoded(line);
        REQUIRE(std::holds_alternative<ai::FinishPart>(part));
        CHECK(std::get<ai::FinishPart>(part).reason == c.received);
    }
}

TEST_CASE("usage is only carried when the totals are known", "[data_stream]") {
    SECTION("both totals present") {
        auto part = decoded(encoded(ai::FinishPart{
            .reason = ai::FinishReason::Stop, .usage = usage_of(11, 22)}));
        REQUIRE(std::holds_alternative<ai::FinishPart>(part));
        auto usage = std::get<ai::FinishPart>(part).usage;
        REQUIRE(usage.input_tokens.total.has_value());
        REQUIRE(usage.output_tokens.total.has_value());
        CHECK(*usage.input_tokens.total == 11);
        CHECK(*usage.output_tokens.total == 22);
    }

    SECTION("a zero total is still transmitted") {
        // Zero is a real measurement, not an absent one; an encoder that tested
        // the count rather than the optional would drop it.
        auto line = encoded(ai::FinishPart{.reason = ai::FinishReason::Stop,
                                           .usage = usage_of(0, 0)});
        CHECK(line.find("\"promptTokens\":0") != std::string::npos);
        CHECK(line.find("\"completionTokens\":0") != std::string::npos);
    }

    SECTION("unknown totals are omitted, not sent as zero") {
        // Reporting "0 prompt tokens" for a provider that did not say is worse
        // than reporting nothing: it silently corrupts cost accounting.
        auto line = encoded(ai::FinishPart{.reason = ai::FinishReason::Stop, .usage = {}});
        CHECK(line.find("promptTokens") == std::string::npos);
        CHECK(line.find("completionTokens") == std::string::npos);

        auto part = decoded(line);
        REQUIRE(std::holds_alternative<ai::FinishPart>(part));
        CHECK_FALSE(std::get<ai::FinishPart>(part).usage.input_tokens.total.has_value());
    }
}

TEST_CASE("parts with no wire representation encode to nothing", "[data_stream]") {
    // The protocol covers text, tool input, finish and error — nothing else. The
    // encoder signals that by returning an empty string, which is
    // indistinguishable from an empty payload, so anything a caller relies on
    // being forwarded has to be checked against this list.
    CHECK(has_no_encoding(ai::StreamStart{}));
    CHECK(has_no_encoding(ai::TextStart{.id = "t"}));
    CHECK(has_no_encoding(ai::TextEnd{.id = "t"}));

    // Reasoning in particular: it is a shipped feature (Anthropic extended
    // thinking, DeepSeek reasoning) and it does not survive this encoder.
    CHECK(has_no_encoding(ai::ReasoningStart{.id = "r"}));
    CHECK(has_no_encoding(ai::ReasoningDelta{.id = "r", .delta = "thinking"}));
    CHECK(has_no_encoding(ai::ReasoningEnd{.id = "r"}));

    CHECK(has_no_encoding(ai::ResponseMetadataPart{}));
    CHECK(has_no_encoding(ai::RawPart{.raw_value = boost::json::value("raw")}));
}

TEST_CASE("the decoder rejects lines that are not protocol frames", "[data_stream]") {
    DataStreamDecoder decoder;

    // Too short to hold a type and a separator.
    CHECK_FALSE(decoder.decode_line("").has_value());
    CHECK_FALSE(decoder.decode_line("0").has_value());

    // A separator is required, and it has to be in the second position.
    CHECK_FALSE(decoder.decode_line("0abc").has_value());
    CHECK_FALSE(decoder.decode_line("00:\"x\"").has_value());

    // Codes the protocol does not define.
    CHECK_FALSE(decoder.decode_line("9:\"x\"").has_value());
    CHECK_FALSE(decoder.decode_line("z:{}").has_value());

    // A recognised code is not enough if the payload is not usable as one.
    CHECK_FALSE(decoder.decode_line("0:").has_value());
}

TEST_CASE("the decoder does not throw on a malformed payload", "[data_stream]") {
    // decode_line returns optional, which reads as "this may fail" rather than
    // "this may throw". It is fed lines off a socket, so malformed input is the
    // expected case, not an exceptional one — a throw here lands in the middle
    // of a streaming loop.
    DataStreamDecoder decoder;
    CHECK_NOTHROW(decoder.decode_line("0:{not json}"));
    CHECK_NOTHROW(decoder.decode_line("1:{}"));
    CHECK_NOTHROW(decoder.decode_line("2:{}"));
    CHECK_NOTHROW(decoder.decode_line("3:{}"));
    CHECK_NOTHROW(decoder.decode_line("0:123"));

    // And when it does fail, it has to say so rather than hand back a part
    // built from partial data.
    CHECK_FALSE(decoder.decode_line("1:{}").has_value());
}
