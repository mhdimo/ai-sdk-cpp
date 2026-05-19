#pragma once

#include <ai/model/provider.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace ai {

class ProviderRegistry {
public:
    static ProviderRegistry& instance() {
        static ProviderRegistry reg;
        return reg;
    }

    void register_provider(std::string_view name, ProviderPtr provider) {
        providers_[std::string(name)] = std::move(provider);
    }

    LanguageModelPtr language_model(std::string_view model_string) {
        auto colon = model_string.find(':');
        if (colon == std::string_view::npos) {
            throw std::runtime_error("Invalid model string format. Expected 'provider:model_id', got: " + std::string(model_string));
        }
        auto provider_name = model_string.substr(0, colon);
        auto model_id = model_string.substr(colon + 1);
        auto p = provider(provider_name);
        if (!p) throw std::runtime_error("Provider not registered: " + std::string(provider_name));
        return p->language_model(model_id);
    }

    ProviderPtr provider(std::string_view name) {
        auto it = providers_.find(std::string(name));
        return it != providers_.end() ? it->second : nullptr;
    }

    bool has_provider(std::string_view name) const {
        return providers_.count(std::string(name)) > 0;
    }

    std::vector<std::string> provider_names() const {
        std::vector<std::string> names;
        for (auto& [k, _] : providers_) names.push_back(k);
        return names;
    }

private:
    ProviderRegistry() = default;
    std::unordered_map<std::string, ProviderPtr> providers_;
};

inline LanguageModelPtr model(std::string_view model_string) {
    return ProviderRegistry::instance().language_model(model_string);
}

} // namespace ai
