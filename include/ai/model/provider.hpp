#pragma once

#include <ai/model/language_model.hpp>
#include <memory>
#include <string>
#include <string_view>

namespace ai {

namespace batch { class BatchProcessor; }

class Provider {
public:
    virtual ~Provider() = default;
    virtual LanguageModelPtr language_model(std::string_view model_id) = 0;
    virtual std::string_view provider_id() const = 0;

    /// Returns a batch processor for this provider + model, or nullptr if the
    /// provider does not support batching. This is the polymorphic seam that lets
    /// core helpers and the C binding reach provider-specific batch processors.
    virtual std::shared_ptr<batch::BatchProcessor> batch_processor(std::string_view /*model_id*/) {
        return nullptr;
    }
};

using ProviderPtr = std::shared_ptr<Provider>;

} // namespace ai
