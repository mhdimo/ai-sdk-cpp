#include <catch2/catch_test_macros.hpp>

#include <ai/tools/standard/standard_toolkit.hpp>
#include <ai/tool/tool.hpp>
#include <ai/tool/tool_set.hpp>

#include <boost/asio.hpp>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

template <typename T>
T run(ai::Task<T> task, boost::asio::io_context& ioc) {
    task.start();
    int guard = 0;
    while (!task.done() && guard++ < 100'000) {
        ioc.run_one();
    }
    REQUIRE(task.done());
    return task.get();
}

fs::path make_unique_temp_dir() {
    static std::atomic<unsigned> counter{0};
    auto dir = fs::temp_directory_path() /
               ("ai-sdk-toolkit-test-" + std::to_string(++counter));
    fs::create_directories(dir);
    return dir;
}

// RAII temp directory.
struct TempDir {
    fs::path path;
    TempDir() : path(make_unique_temp_dir()) {}
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

void write_file(const fs::path& p, const std::string& content) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f << content;
}

std::string read_back(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

boost::json::value run_tool(const ai::ToolSet& tools, const std::string& name,
                            const boost::json::value& input,
                            boost::asio::io_context& ioc) {
    const auto* def = tools.find(name);
    REQUIRE(def);
    REQUIRE(def->execute);
    return run((*def->execute)(input, ai::ToolExecutionContext{}), ioc);
}

} // namespace

// ---------------------------------------------------------------------------
// edit_file
// ---------------------------------------------------------------------------

TEST_CASE("edit_file replaces a unique exact match", "[toolkit]") {
    boost::asio::io_context ioc;
    TempDir tmp;
    fs::path file = tmp.path / "f.txt";
    write_file(file, "alpha\nbeta\ngamma\n");

    auto out = run_tool(ai::standard_toolkit(), "edit_file",
        boost::json::object{{"path", file.string()}, {"old_string", "beta"},
                             {"new_string", "BETA"}}, ioc);

    REQUIRE(out.is_string());
    REQUIRE(read_back(file) == "alpha\nBETA\ngamma\n");
}

TEST_CASE("edit_file rejects an ambiguous match without replace_all", "[toolkit]") {
    boost::asio::io_context ioc;
    TempDir tmp;
    fs::path file = tmp.path / "f.txt";
    write_file(file, "x x x");

    REQUIRE_THROWS_AS(run_tool(ai::standard_toolkit(), "edit_file",
        boost::json::object{{"path", file.string()}, {"old_string", "x"},
                             {"new_string", "y"}}, ioc),
        std::runtime_error);
    // File unchanged on failure.
    REQUIRE(read_back(file) == "x x x");
}

TEST_CASE("edit_file replace_all replaces every occurrence", "[toolkit]") {
    boost::asio::io_context ioc;
    TempDir tmp;
    fs::path file = tmp.path / "f.txt";
    write_file(file, "x x x");

    auto out = run_tool(ai::standard_toolkit(), "edit_file",
        boost::json::object{{"path", file.string()}, {"old_string", "x"},
                             {"new_string", "y"}, {"replace_all", true}}, ioc);

    REQUIRE(out.as_string().find("3 replacement") != std::string::npos);
    REQUIRE(read_back(file) == "y y y");
}

TEST_CASE("edit_file throws when old_string is absent", "[toolkit]") {
    boost::asio::io_context ioc;
    TempDir tmp;
    fs::path file = tmp.path / "f.txt";
    write_file(file, "hello");

    REQUIRE_THROWS_AS(run_tool(ai::standard_toolkit(), "edit_file",
        boost::json::object{{"path", file.string()}, {"old_string", "missing"},
                             {"new_string", "z"}}, ioc),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// read_file / write_file
// ---------------------------------------------------------------------------

TEST_CASE("read_file returns contents with offset/limit", "[toolkit]") {
    boost::asio::io_context ioc;
    TempDir tmp;
    fs::path file = tmp.path / "f.txt";
    write_file(file, "line1\nline2\nline3\nline4\n");

    auto out = run_tool(ai::standard_toolkit(), "read_file",
        boost::json::object{{"path", file.string()}, {"offset", 2}, {"limit", 1}}, ioc);
    REQUIRE(out.as_string() == "line2\n");
}

TEST_CASE("write_file creates parent directories and writes content", "[toolkit]") {
    boost::asio::io_context ioc;
    TempDir tmp;
    fs::path file = tmp.path / "a" / "b" / "c.txt";

    auto out = run_tool(ai::standard_toolkit(), "write_file",
        boost::json::object{{"path", file.string()}, {"content", "payload"}}, ioc);
    REQUIRE(out.as_string().find("Wrote 7 bytes") != std::string::npos);
    REQUIRE(fs::exists(file));
    REQUIRE(read_back(file) == "payload");
}

// ---------------------------------------------------------------------------
// glob
// ---------------------------------------------------------------------------

TEST_CASE("glob matches files by pattern", "[toolkit]") {
    boost::asio::io_context ioc;
    TempDir tmp;
    write_file(tmp.path / "a.txt", "x");
    write_file(tmp.path / "sub" / "b.txt", "x");
    write_file(tmp.path / "c.md", "x");

    auto out = run_tool(ai::standard_toolkit(), "glob",
        boost::json::object{{"pattern", "**/*.txt"}, {"path", tmp.path.string()}}, ioc);

    REQUIRE(out.is_array());
    REQUIRE(out.as_array().size() == 2);
}

// ---------------------------------------------------------------------------
// grep
// ---------------------------------------------------------------------------

TEST_CASE("grep finds matching lines", "[toolkit]") {
    boost::asio::io_context ioc;
    TempDir tmp;
    fs::path file = tmp.path / "f.txt";
    write_file(file, "alpha\nbeta\ngamma\n");

    auto out = run_tool(ai::standard_toolkit(), "grep",
        boost::json::object{{"pattern", "beta"}, {"path", file.string()}}, ioc);

    REQUIRE(out.is_array());
    REQUIRE(out.as_array().size() == 1);
    REQUIRE(out.as_array()[0].as_string().find("beta") != std::string::npos);
}

// ---------------------------------------------------------------------------
// bash
// ---------------------------------------------------------------------------

TEST_CASE("bash captures command output", "[toolkit]") {
    boost::asio::io_context ioc;
    auto out = run_tool(ai::standard_toolkit(), "bash",
        boost::json::object{{"command", "echo hello_toolkit"}}, ioc);
    REQUIRE(out.as_string().find("hello_toolkit") != std::string::npos);
}
