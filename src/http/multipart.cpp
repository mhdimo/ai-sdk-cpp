#include <ai/http/multipart.hpp>
#include <random>
#include <sstream>
#include <iomanip>

namespace ai::http {

static std::string random_hex(size_t length) {
    static const char hex_chars[] = "0123456789abcdef";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 15);

    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        result += hex_chars[dist(gen)];
    }
    return result;
}

std::string MultipartFormData::generate_boundary() {
    return "----AiSdkBoundary" + random_hex(24);
}

MultipartFormData::MultipartFormData()
    : boundary_(generate_boundary()) {}

void MultipartFormData::add_field(std::string name, std::string value) {
    fields_.push_back(Field{std::move(name), std::move(value)});
}

void MultipartFormData::add_file(std::string name, std::string filename,
                                  std::string content_type, std::vector<uint8_t> data) {
    files_.push_back(File{
        std::move(name), std::move(filename),
        std::move(content_type), std::move(data)
    });
}

std::string MultipartFormData::content_type() const {
    return "multipart/form-data; boundary=" + boundary_;
}

std::string MultipartFormData::body() const {
    std::string result;

    for (const auto& field : fields_) {
        result += "--" + boundary_ + "\r\n";
        result += "Content-Disposition: form-data; name=\"" + field.name + "\"\r\n";
        result += "\r\n";
        result += field.value + "\r\n";
    }

    for (const auto& file : files_) {
        result += "--" + boundary_ + "\r\n";
        result += "Content-Disposition: form-data; name=\"" + file.name + "\"; filename=\"" + file.filename + "\"\r\n";
        result += "Content-Type: " + file.content_type + "\r\n";
        result += "\r\n";
        result.append(reinterpret_cast<const char*>(file.data.data()), file.data.size());
        result += "\r\n";
    }

    result += "--" + boundary_ + "--\r\n";

    return result;
}

} // namespace ai::http
