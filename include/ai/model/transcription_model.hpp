#pragma once

#include <ai/stream/async_generator.hpp>
#include <ai/stream/stream_part.hpp>
#include <boost/json.hpp>
#include <string>
#include <vector>
#include <optional>
#include <memory>

namespace ai {

struct TranscriptionSegment {
    std::string text;
    double start;
    double end;
};

struct TranscriptionModelOptions {
    std::vector<uint8_t> audio;
    std::string media_type;
    std::optional<std::string> language;
    std::optional<std::string> prompt;
    boost::json::object provider_options;
};

struct TranscriptionModelResult {
    std::string text;
    std::vector<TranscriptionSegment> segments;
    std::optional<std::string> language;
    Usage usage;
};

class TranscriptionModel {
public:
    virtual ~TranscriptionModel() = default;
    virtual std::string_view provider() const = 0;
    virtual std::string_view model_id() const = 0;
    virtual Task<TranscriptionModelResult> do_transcribe(TranscriptionModelOptions options) = 0;
};

using TranscriptionModelPtr = std::shared_ptr<TranscriptionModel>;

} // namespace ai
