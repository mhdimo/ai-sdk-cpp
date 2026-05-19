#pragma once

#include <ai/model/language_model.hpp>
#include <ai/model/call_options.hpp>
#include <ai/model/generate_result.hpp>
#include <ai/stream/async_generator.hpp>
#include <ai/util/cancellation.hpp>
#include <boost/json.hpp>
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <memory>

namespace ai::batch {

struct BatchRequest {
    std::string custom_id;  // user-defined ID to match responses
    CallOptions options;    // the actual model call parameters
};

enum class BatchStatus {
    Validating,
    InProgress,
    Completed,
    Failed,
    Expired,
    Cancelling,
    Cancelled
};

struct BatchResponseItem {
    std::string custom_id;
    std::optional<GenerateResult> result;
    std::optional<std::string> error;
};

struct BatchInfo {
    std::string id;
    BatchStatus status;
    int total_requests = 0;
    int completed_requests = 0;
    int failed_requests = 0;
    std::optional<std::string> created_at;
    std::optional<std::string> completed_at;
};

class BatchProcessor {
public:
    virtual ~BatchProcessor() = default;

    // Submit a batch of requests, get back a batch ID
    virtual Task<std::string> submit(std::vector<BatchRequest> requests) = 0;

    // Check batch status
    virtual Task<BatchInfo> status(std::string_view batch_id) = 0;

    // Retrieve results (only available when status == Completed)
    virtual Task<std::vector<BatchResponseItem>> results(std::string_view batch_id) = 0;

    // Cancel a batch
    virtual Task<void> cancel(std::string_view batch_id) = 0;

    // List batches
    virtual Task<std::vector<BatchInfo>> list(int limit = 20) = 0;
};

using BatchProcessorPtr = std::shared_ptr<BatchProcessor>;

} // namespace ai::batch
