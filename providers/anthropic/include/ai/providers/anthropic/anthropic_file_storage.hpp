#pragma once

#include <ai/model/file_storage.hpp>
#include <ai/http/client.hpp>
#include <string>

namespace ai::providers::anthropic {

class AnthropicProvider;

class AnthropicFileStorage : public FileStorage {
public:
    explicit AnthropicFileStorage(AnthropicProvider& provider);

    std::string_view provider() const override { return "anthropic"; }
    Task<FileUploadResult> upload(FileUploadOptions options) override;

private:
    AnthropicProvider& provider_;
};

} // namespace ai::providers::anthropic
