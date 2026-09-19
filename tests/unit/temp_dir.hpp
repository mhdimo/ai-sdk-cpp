#pragma once

// A scratch directory that is unique for the life of one *process*, not merely
// one test case.
//
// The distinction is the whole reason this is a shared header. Every test file
// used to name its directory from a counter in a function-local static, which
// is per-process -- and `catch_discover_tests()` registers each TEST_CASE as a
// separate ctest entry, so `ctest -j` runs them as separate processes. Every
// process therefore handed out the same first few names: two tests running side
// by side picked the same directory, and whichever finished first deleted it
// while the other was still writing to it. That shows up as an intermittent
// failure of whichever test lost the race, with nothing in the output to say
// why -- "MarkdownMemoryStore round-trips records with fidelity" failed under
// `ctest -j` and passed on its own, which is what led here.
//
// The pid makes the name unique across processes; the counter keeps it unique
// within one. Both are needed.

#include <atomic>
#include <filesystem>
#include <string>
#include <system_error>

#if defined(_WIN32)
#include <process.h>
#define AI_SDK_TEST_GETPID _getpid
#else
#include <unistd.h>
#define AI_SDK_TEST_GETPID getpid
#endif

namespace ai::test {

inline std::filesystem::path make_unique_temp_dir(const std::string& prefix) {
    static std::atomic<unsigned> counter{0};
    auto dir = std::filesystem::temp_directory_path() /
               (prefix + std::to_string(AI_SDK_TEST_GETPID()) + "-" +
                std::to_string(++counter));
    std::filesystem::create_directories(dir);
    return dir;
}

/// Removes its directory when it goes out of scope.
struct TempDir {
    std::filesystem::path path;

    explicit TempDir(const std::string& prefix = "ai-sdk-test-")
        : path(make_unique_temp_dir(prefix)) {}

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
};

} // namespace ai::test
