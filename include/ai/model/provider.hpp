#pragma once

#include <ai/model/language_model.hpp>
#include <memory>
#include <string>
#include <string_view>

namespace ai {

class Provider {
public:
    virtual ~Provider() = default;
    virtual LanguageModelPtr language_model(std::string_view model_id) = 0;
    virtual std::string_view provider_id() const = 0;
};

using ProviderPtr = std::shared_ptr<Provider>;

} // namespace ai
