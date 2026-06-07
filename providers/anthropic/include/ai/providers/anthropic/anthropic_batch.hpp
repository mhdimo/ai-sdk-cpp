#pragma once

#include <ai/core/batch.hpp>
#include <ai/providers/anthropic/anthropic.hpp>
#include <string>
#include <vector>

namespace ai::providers::anthropic {

class AnthropicBatchProcessor : public ai::batch::BatchProcessor {
public:
    AnthropicBatchProcessor(AnthropicProvider& provider, std::string model_id);

    Task<std::string> submit(std::vector<ai::batch::BatchRequest> requests) override;
    Task<ai::batch::BatchInfo> status(std::string_view batch_id) override;
    Task<std::vector<ai::batch::BatchResponseItem>> results(std::string_view batch_id) override;
    Task<void> cancel(std::string_view batch_id) override;
    Task<std::vector<ai::batch::BatchInfo>> list(int limit = 20) override;

private:
    AnthropicProvider& provider_;
    std::string model_id_;

    // Build a messages request body from CallOptions
    boost::json::object build_messages_params(const CallOptions& options);

    // Parse a single batch result line into a BatchResponseItem
    ai::batch::BatchResponseItem parse_result_line(const boost::json::value& line_val);

    // Parse batch info from API response JSON
    ai::batch::BatchInfo parse_batch_info(const boost::json::value& val);
};

} // namespace ai::providers::anthropic
