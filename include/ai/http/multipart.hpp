#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace ai::http {

class MultipartFormData {
public:
    MultipartFormData();

    void add_field(std::string name, std::string value);
    void add_file(std::string name, std::string filename,
                  std::string content_type, std::vector<uint8_t> data);

    std::string content_type() const;
    std::string body() const;

private:
    struct Field {
        std::string name;
        std::string value;
    };

    struct File {
        std::string name;
        std::string filename;
        std::string content_type;
        std::vector<uint8_t> data;
    };

    std::string boundary_;
    std::vector<Field> fields_;
    std::vector<File> files_;

    static std::string generate_boundary();
};

} // namespace ai::http
