#include <ai/ai.hpp>
#include <ai/providers/deepseek/deepseek.hpp>
#include <boost/asio.hpp>
#include <iostream>

int main() {
    boost::asio::io_context ioc;

    auto deepseek = ai::providers::deepseek::create_deepseek({
        .io_context = ioc,
    });

    auto model = deepseek->language_model("deepseek-v4-flash");

    auto task = ai::generate_text({
        .model = model,
        .prompt = "Explain C++20 coroutines in 3 sentences.",
        .max_output_tokens = 256,
        .temperature = 0.7,
    });

    task.start();
    while (!task.done()) {
        ioc.run_one();
    }

    try {
        auto result = task.get();
        std::cout << result.text << "\n";
        std::cout << "\n[Tokens: " << result.usage.input_tokens.total.value_or(0)
                  << " in, " << result.usage.output_tokens.total.value_or(0) << " out]\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
