#include <ai/ai.hpp>
#include <ai/providers/deepseek/deepseek.hpp>
#include <boost/asio.hpp>
#include <iostream>

int main() {
    try {
        std::cout << "Creating io_context...\n";
        boost::asio::io_context ioc;
        
        std::cout << "Creating DeepSeek provider...\n";
        auto deepseek = ai::providers::deepseek::create_deepseek({
            .io_context = ioc,
        });
        
        std::cout << "Creating language model...\n";
        auto model = deepseek->language_model("deepseek-v4-flash");
        
        std::cout << "Creating generate task...\n";
        auto task = ai::generate_text({
            .model = model,
            .prompt = "Hello",
            .max_output_tokens = 256,
        });

        std::cout << "Starting task...\n";
        task.start();
        
        std::cout << "Running io_context...\n";
        while (!task.done()) {
            ioc.run_one();
        }
        
        std::cout << "Getting result...\n";
        auto result = task.get();
        std::cout << result.text << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
