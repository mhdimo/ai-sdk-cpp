// A minimal Claude-Code-style coding agent built on the CodingAgent facade.
// The facade bundles a context-managed Session, the standard toolkit
// (read_file/write_file/edit_file/glob/grep/bash), optional permission gating,
// and optional persistent memory — so a real coding agent is ~30 lines.

#include <ai/ai.hpp>
#include <ai/providers/deepseek/deepseek.hpp>

#include <boost/asio.hpp>

#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

int main() {
    boost::asio::io_context ioc;

    auto deepseek = ai::providers::deepseek::create_deepseek({.io_context = ioc});
    auto model = deepseek->language_model("deepseek-v4-flash");

    ai::CodingAgentOptions opts;
    opts.model = model;
    opts.instructions =
        "You are an expert coding assistant, like Claude Code. You can read, "
        "edit, and search files and run shell commands. Make targeted, minimal "
        "changes and explain what you're doing. Current working directory: " +
        fs::current_path().string();
    opts.max_steps = 50;
    // opts.enable_memory = true;  // opt-in cross-session memory (.agent/memory)

    ai::CodingAgent agent(opts);

    std::cout << "╔══════════════════════════════════════════╗\n"
              << "║  AI Coding Agent (ai-sdk-cpp)            ║\n"
              << "║  empty line = send · Ctrl-D = exit       ║\n"
              << "╚══════════════════════════════════════════╝\n\n";

    std::string line;
    std::string buf;
    std::cout << "› " << std::flush;
    while (std::getline(std::cin, line)) {
        if (line.empty() && !buf.empty()) {
            auto task = agent.send(buf);
            task.start();
            while (!task.done()) {
                ioc.run_one();
            }
            try {
                auto result = task.get();
                std::cout << "\n" << result.text
                          << "\n━━━ " << agent.session().metadata().turns << " turns ━━━\n";
            } catch (const std::exception& e) {
                std::cerr << "\nError: " << e.what() << "\n";
            }
            buf.clear();
            std::cout << "\n› " << std::flush;
        } else if (!line.empty()) {
            if (!buf.empty()) buf += '\n';
            buf += line;
            std::cout << "  " << std::flush;
        }
    }
    std::cout << "\n";
    return 0;
}
