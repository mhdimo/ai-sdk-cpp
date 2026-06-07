#include <ai/ai.hpp>
#include <ai/providers/anthropic/anthropic.hpp>
#include <boost/asio.hpp>
#include <iostream>

int main() {
    boost::asio::io_context ioc;

    auto anthropic = ai::providers::anthropic::create_anthropic({
        .io_context = ioc,
    });

    auto model = anthropic->language_model("claude-sonnet-4-20250514");

    auto task = ai::stream_text({
        .model = model,
        .prompt = "Write a haiku about C++ templates.",
        .max_output_tokens = 128,
    });

    task.start();
    while (!task.done()) {
        ioc.run_one();
    }

    try {
        auto result = task.get();
        auto& stream = result.stream;

        // Consume the stream
        while (true) {
            auto next = stream.next();
            // In a real async context, we'd co_await this
            // For this example, drive the coroutine manually
            break; // placeholder - real usage requires async context
        }

        std::cout << "\n[Stream complete]\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
