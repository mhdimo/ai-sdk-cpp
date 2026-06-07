#pragma once

#include <ai/core/batch.hpp>
#include <ai/providers/openai/openai.hpp>
#include <string>
#include <vector>

namespace ai::providers::openai {

class OpenAIBatchProcessor : public ai::batch::BatchProcessor {
public:
    OpenAIBatchProcessor(OpenAIProvider& provider, std::string model_id);

    Task<std::string> submit(std::vector<ai::batch::BatchRequest> requests) override;
    Task<ai::batch::BatchInfo> status(std::string_view batch_id) override;
    Task<std::vector<ai::batch::BatchResponseItem>> results(std::string_view batch_id) override;
    Task<void> cancel(std::string_view batch_id) override;
    Task<std::vector<ai::batch::BatchInfo>> list(int limit = 20) override;

private:
    OpenAIProvider& provider_;
    std::string model_id_;

    // Convert CallOptions into a chat completions request body
    boost::json::object build_chat_body(const CallOptions& options);

    // Parse a single batch response line into a BatchResponseItem
    ai::batch::BatchResponseItem parse_response_line(const boost::json::value& line_val);

    // Parse batch info from API response JSON
    ai::batch::BatchInfo parse_batch_info(const boost::json::value& val);
};

} // namespace ai::providers::openai
