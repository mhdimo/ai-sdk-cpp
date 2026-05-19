#pragma once

#include <ai/stream/async_generator.hpp>
#include <ai/util/cancellation.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>
#include <optional>

namespace ai {

struct FileUploadOptions {
    std::string filename;
    std::string content_type;
    std::vector<uint8_t> data;
    std::optional<std::string> purpose;
    CancellationToken cancel;
};

struct FileUploadResult {
    std::string file_id;
    std::string filename;
    int bytes = 0;
    std::optional<std::string> purpose;
    std::optional<std::string> status;
};

class FileStorage {
public:
    virtual ~FileStorage() = default;

    virtual std::string_view provider() const = 0;
    virtual Task<FileUploadResult> upload(FileUploadOptions options) = 0;
};

using FileStoragePtr = std::shared_ptr<FileStorage>;

} // namespace ai
