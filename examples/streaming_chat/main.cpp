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

    // stream_text returns a Task whose result holds the stream we then drain.
    auto outer = ai::stream_text({
        .model = model,
        .prompt = "Write a haiku about C++ templates.",
        .max_output_tokens = 128,
    });

    outer.start();
    while (!outer.done()) {
        ioc.run_one();
    }

    try {
        auto result = outer.get();

        // The stream is an AsyncGenerator: generator.next() must be awaited from
        // inside a coroutine. Drain it in a second Task driven on the same
        // io_context, printing each text delta as it arrives.
        auto consume = [](ai::AsyncGenerator<ai::StreamPart> stream) -> ai::Task<void> {
            while (auto part = co_await stream.next()) {
                if (auto* delta = std::get_if<ai::TextDelta>(&*part)) {
                    std::cout << delta->delta << std::flush;
                }
            }
        }(std::move(result.stream));

        consume.start();
        while (!consume.done()) {
            ioc.run_one();
        }
        consume.get(); // rethrows any stream error

        std::cout << "\n[Stream complete]\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
