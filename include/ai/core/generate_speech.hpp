#pragma once

#include <ai/model/speech_model.hpp>
#include <ai/util/cancellation.hpp>
#include <string>
#include <optional>

namespace ai {

struct GenerateSpeechOptions {
    SpeechModelPtr model;
    std::string text;
    std::optional<std::string> voice;
    std::optional<double> speed;
    std::optional<std::string> format;
    CancellationToken cancel;
    boost::json::object provider_options;
};

struct GenerateSpeechResult {
    std::vector<uint8_t> audio;
    std::string media_type;
    Usage usage;
};

Task<GenerateSpeechResult> generate_speech(GenerateSpeechOptions options);

} // namespace ai
