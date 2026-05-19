#pragma once

#include <ai/stream/async_generator.hpp>
#include <ai/stream/stream_part.hpp>
#include <boost/json.hpp>
#include <string>
#include <vector>
#include <optional>
#include <memory>

namespace ai {

struct SpeechModelOptions {
    std::string text;
    std::optional<std::string> voice;
    std::optional<double> speed;
    std::optional<std::string> format;
    boost::json::object provider_options;
};

struct SpeechModelResult {
    std::vector<uint8_t> audio;
    std::string media_type;
    Usage usage;
};

class SpeechModel {
public:
    virtual ~SpeechModel() = default;
    virtual std::string_view provider() const = 0;
    virtual std::string_view model_id() const = 0;
    virtual Task<SpeechModelResult> do_generate(SpeechModelOptions options) = 0;
};

using SpeechModelPtr = std::shared_ptr<SpeechModel>;

} // namespace ai
