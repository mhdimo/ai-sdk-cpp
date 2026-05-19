#include <ai/ai.hpp>
#include <ai/providers/bedrock/bedrock.hpp>
#include <boost/asio.hpp>
#include <iostream>

int main() {
    try {
        boost::asio::io_context ioc;
        std::cout << "Creating provider..." << std::endl;

        auto provider = ai::providers::bedrock::create_bedrock({
            .io_context = ioc,
        });
        std::cout << "Provider created. Base URL: " << std::flush;

        auto* bp = dynamic_cast<ai::providers::bedrock::BedrockProvider*>(provider.get());
        if (bp) std::cout << bp->runtime_base_url() << std::endl;

        std::cout << "Creating model..." << std::endl;
        auto model = provider->language_model("us.anthropic.claude-opus-4-6-v1");
        std::cout << "Model created." << std::endl;

        std::cout << "Calling generate_text..." << std::endl;
        auto task = ai::generate_text({
            .model = model,
            .prompt = "Say hello in exactly one sentence.",
            .max_output_tokens = 50,
        });

        task.start();
        std::cout << "Task started, running event loop..." << std::endl;

        while (!task.done()) {
            ioc.run_one();
        }

        auto result = task.get();
        std::cout << "Result: " << result.text << std::endl;
        std::cout << "Tokens: " << result.usage.input_tokens.total.value_or(0)
                  << " in, " << result.usage.output_tokens.total.value_or(0) << " out" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
