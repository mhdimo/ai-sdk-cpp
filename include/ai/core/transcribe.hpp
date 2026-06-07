#pragma once

#include <ai/model/transcription_model.hpp>
#include <ai/util/cancellation.hpp>
#include <string>
#include <vector>
#include <optional>

namespace ai {

struct TranscribeOptions {
    TranscriptionModelPtr model;
    std::vector<uint8_t> audio;
    std::string media_type;
    std::optional<std::string> language;
    std::optional<std::string> prompt;
    CancellationToken cancel;
    boost::json::object provider_options;
};

struct TranscribeResult {
    std::string text;
    std::vector<TranscriptionSegment> segments;
    std::optional<std::string> language;
    Usage usage;
};

Task<TranscribeResult> transcribe(TranscribeOptions options);

} // namespace ai
