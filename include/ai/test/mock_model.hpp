#pragma once

#include <ai/model/language_model.hpp>
#include <ai/model/call_options.hpp>
#include <ai/model/generate_result.hpp>
#include <ai/model/stream_result.hpp>
#include <ai/stream/async_generator.hpp>
#include <ai/stream/stream_part.hpp>
#include <vector>
#include <stdexcept>

namespace ai::test {

struct MockResponse {
    std::string text;
    std::vector<ToolCallContent> tool_calls;
    FinishReason finish_reason = FinishReason::Stop;
    Usage usage = {};
};

class MockLanguageModel : public LanguageModel {
public:
    explicit MockLanguageModel(std::string model_id = "mock-model")
        : model_id_(std::move(model_id)) {}

    void queue_response(MockResponse response) {
        responses_.push_back(std::move(response));
    }

    void queue_text(std::string text) {
        responses_.push_back(MockResponse{.text = std::move(text)});
    }

    void queue_tool_call(std::string tool_name, std::string input_json) {
        MockResponse r;
        r.tool_calls.push_back(ToolCallContent{
            .tool_call_id = "call_" + std::to_string(response_index_),
            .tool_name = std::move(tool_name),
            .input = std::move(input_json)
        });
        r.finish_reason = FinishReason::ToolCalls;
        responses_.push_back(std::move(r));
    }

    int call_count() const { return call_count_; }
    const std::vector<CallOptions>& received_options() const { return received_; }
    const CallOptions& last_options() const { return received_.back(); }

    std::string_view provider() const override { return "mock"; }
    std::string_view model_id() const override { return model_id_; }

    Task<GenerateResult> do_generate(CallOptions options) override {
        received_.push_back(std::move(options));
        call_count_++;

        if (response_index_ >= (int)responses_.size()) {
            throw std::runtime_error("MockLanguageModel: no more queued responses");
        }

        auto& resp = responses_[response_index_++];
        GenerateResult result;
        result.finish_reason = resp.finish_reason;
        result.usage = resp.usage;

        if (!resp.text.empty()) {
            result.content.push_back(TextContent{resp.text});
        }
        for (auto& tc : resp.tool_calls) {
            result.content.push_back(tc);
        }

        co_return result;
    }

    Task<StreamResult> do_stream(CallOptions options) override {
        received_.push_back(std::move(options));
        call_count_++;

        if (response_index_ >= (int)responses_.size()) {
            throw std::runtime_error("MockLanguageModel: no more queued responses");
        }

        auto& resp = responses_[response_index_++];
        auto text = resp.text;
        auto finish = resp.finish_reason;
        auto usage = resp.usage;

        auto gen = [](std::string t, FinishReason fr, Usage u) -> AsyncGenerator<StreamPart> {
            if (!t.empty()) {
                co_yield TextStart{.id = "0"};
                co_yield TextDelta{.id = "0", .delta = t};
                co_yield TextEnd{.id = "0"};
            }
            co_yield FinishPart{.reason = fr, .usage = u};
        }(std::move(text), finish, usage);

        co_return StreamResult{.stream = std::move(gen)};
    }

private:
    std::string model_id_;
    std::vector<MockResponse> responses_;
    int response_index_ = 0;
    int call_count_ = 0;
    std::vector<CallOptions> received_;
};

} // namespace ai::test
