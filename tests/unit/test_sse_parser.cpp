#include <catch2/catch_test_macros.hpp>
#include <ai/stream/sse_parser.hpp>

using namespace ai::stream;

TEST_CASE("SseParser parses basic event", "[sse]") {
    SseParser parser;
    parser.feed("data: hello world\n\n");

    auto event = parser.next_event();
    REQUIRE(event.has_value());
    REQUIRE(event->data == "hello world");
    REQUIRE(event->event == "message");
}

TEST_CASE("SseParser parses named event", "[sse]") {
    SseParser parser;
    parser.feed("event: custom\ndata: payload\n\n");

    auto event = parser.next_event();
    REQUIRE(event.has_value());
    REQUIRE(event->event == "custom");
    REQUIRE(event->data == "payload");
}

TEST_CASE("SseParser handles multi-line data", "[sse]") {
    SseParser parser;
    parser.feed("data: line1\ndata: line2\ndata: line3\n\n");

    auto event = parser.next_event();
    REQUIRE(event.has_value());
    REQUIRE(event->data == "line1\nline2\nline3");
}

TEST_CASE("SseParser skips comments", "[sse]") {
    SseParser parser;
    parser.feed(": this is a comment\ndata: actual data\n\n");

    auto event = parser.next_event();
    REQUIRE(event.has_value());
    REQUIRE(event->data == "actual data");
}

TEST_CASE("SseParser handles partial chunks", "[sse]") {
    SseParser parser;
    parser.feed("data: hel");
    REQUIRE_FALSE(parser.next_event().has_value());

    parser.feed("lo\n\n");
    auto event = parser.next_event();
    REQUIRE(event.has_value());
    REQUIRE(event->data == "hello");
}

TEST_CASE("SseParser handles multiple events", "[sse]") {
    SseParser parser;
    parser.feed("data: first\n\ndata: second\n\n");

    auto e1 = parser.next_event();
    REQUIRE(e1.has_value());
    REQUIRE(e1->data == "first");

    auto e2 = parser.next_event();
    REQUIRE(e2.has_value());
    REQUIRE(e2->data == "second");
}

TEST_CASE("SseParser handles id field", "[sse]") {
    SseParser parser;
    parser.feed("id: 42\ndata: test\n\n");

    auto event = parser.next_event();
    REQUIRE(event.has_value());
    REQUIRE(event->id == "42");
    REQUIRE(event->data == "test");
}

TEST_CASE("SseParser handles CRLF line endings", "[sse]") {
    SseParser parser;
    parser.feed("data: hello\r\n\r\n");

    auto event = parser.next_event();
    REQUIRE(event.has_value());
    REQUIRE(event->data == "hello");
}

TEST_CASE("SseParser ignores empty data dispatches", "[sse]") {
    SseParser parser;
    parser.feed("\n\n");  // empty event, no data field

    auto event = parser.next_event();
    REQUIRE_FALSE(event.has_value());
}
