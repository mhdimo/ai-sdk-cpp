#include <catch2/catch_test_macros.hpp>
#include <ai/prompt/message.hpp>

using namespace ai;

TEST_CASE("SystemMessage construction", "[message]") {
    SystemMessage msg{.content = "You are a helpful assistant."};
    REQUIRE(msg.content == "You are a helpful assistant.");
}

TEST_CASE("UserMessage with text content", "[message]") {
    UserContent content;
    content.push_back(TextPart{.text = "Hello, world!"});

    UserMessage msg{.content = std::move(content)};
    REQUIRE(msg.content.size() == 1);

    auto* text = std::get_if<TextPart>(&msg.content[0]);
    REQUIRE(text != nullptr);
    REQUIRE(text->text == "Hello, world!");
}

TEST_CASE("AssistantMessage with tool call", "[message]") {
    AssistantContent content;
    content.push_back(TextPart{.text = "Let me check that."});
    content.push_back(ToolCallPart{
        .tool_call_id = "call_123",
        .tool_name = "get_weather",
        .input = boost::json::value(boost::json::object{{"city", "Paris"}}),
    });

    AssistantMessage msg{.content = std::move(content)};
    REQUIRE(msg.content.size() == 2);

    auto* tc = std::get_if<ToolCallPart>(&msg.content[1]);
    REQUIRE(tc != nullptr);
    REQUIRE(tc->tool_name == "get_weather");
    REQUIRE(tc->tool_call_id == "call_123");
}

TEST_CASE("Message variant holds different types", "[message]") {
    Prompt prompt;
    prompt.push_back(SystemMessage{.content = "System"});

    UserContent user_content;
    user_content.push_back(TextPart{.text = "Hi"});
    prompt.push_back(UserMessage{.content = std::move(user_content)});

    REQUIRE(prompt.size() == 2);
    REQUIRE(std::holds_alternative<SystemMessage>(prompt[0]));
    REQUIRE(std::holds_alternative<UserMessage>(prompt[1]));
}

TEST_CASE("ToolMessage with results", "[message]") {
    ToolContent content;
    content.push_back(ToolResultPart{
        .tool_call_id = "call_456",
        .tool_name = "calculate",
        .output = JsonOutput{.value = boost::json::value(42)},
    });

    ToolMessage msg{.content = std::move(content)};
    REQUIRE(msg.content.size() == 1);
    REQUIRE(msg.content[0].tool_call_id == "call_456");

    auto* json_out = std::get_if<JsonOutput>(&msg.content[0].output);
    REQUIRE(json_out != nullptr);
    REQUIRE(json_out->value.as_int64() == 42);
}
