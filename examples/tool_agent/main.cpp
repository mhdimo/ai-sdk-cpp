#include <ai/ai.hpp>
#include <ai/providers/anthropic/anthropic.hpp>
#include <boost/asio.hpp>
#include <iostream>
#include <cmath>

int main() {
    boost::asio::io_context ioc;

    auto anthropic = ai::providers::anthropic::create_anthropic({
        .io_context = ioc,
    });

    auto model = anthropic->language_model("claude-sonnet-4-20250514");

    ai::ToolSet tools;

    tools.add(ai::tool("calculate",
        ai::schema::JsonSchema::object({
            {"expression", ai::schema::JsonSchema::string("Math expression to evaluate")},
            {"operation", ai::schema::JsonSchema::enum_of({"add", "subtract", "multiply", "divide", "sqrt", "pow"})},
            {"a", ai::schema::JsonSchema::number("First operand")},
            {"b", ai::schema::JsonSchema::number("Second operand (optional for sqrt)")},
        }).required({"operation", "a"}),
        "Perform a mathematical calculation",
        [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto op = std::string(input.at("operation").as_string());
            double a = input.at("a").as_double();
            double b = 0;
            if (auto b_it = input.as_object().find("b"); b_it != input.as_object().end()) {
                if (b_it->value().is_double()) b = b_it->value().as_double();
                else if (b_it->value().is_int64()) b = static_cast<double>(b_it->value().as_int64());
            }

            double result;
            if (op == "add") result = a + b;
            else if (op == "subtract") result = a - b;
            else if (op == "multiply") result = a * b;
            else if (op == "divide") result = b != 0 ? a / b : 0;
            else if (op == "sqrt") result = std::sqrt(a);
            else if (op == "pow") result = std::pow(a, b);
            else co_return boost::json::value("Unknown operation: " + op);

            co_return boost::json::value(result);
        }
    ));

    tools.add(ai::tool("get_weather",
        ai::schema::JsonSchema::object({
            {"city", ai::schema::JsonSchema::string("City name")}
        }).required({"city"}).additional_properties(false),
        "Get the current weather for a city (simulated)",
        [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto city = std::string(input.at("city").as_string());
            boost::json::object weather;
            weather["city"] = city;
            weather["temperature"] = 22;
            weather["unit"] = "celsius";
            weather["condition"] = "sunny";
            co_return boost::json::value(std::move(weather));
        }
    ));

    ai::ToolLoopAgent agent({
        .model = model,
        .tools = std::move(tools),
        .instructions = "You are a helpful assistant with access to a calculator and "
                        "weather service. Use tools when needed to answer questions.",
        .max_steps = 10,
        .on_step_finish = [](const ai::StepResult& step) {
            for (auto& tr : step.tool_results) {
                std::cout << "  [" << tr.tool_name << "] → "
                          << boost::json::serialize(tr.output) << "\n";
            }
        },
    });

    auto task = agent.call(
        "What's the weather in Paris? Also, what is the square root of 144 plus 25 squared?"
    );

    task.start();
    while (!task.done()) {
        ioc.run_one();
    }

    try {
        auto result = task.get();
        std::cout << "\nAgent response:\n" << result.text << "\n";
        std::cout << "\n[Steps: " << result.steps.size()
                  << ", Tool calls: " << result.tool_calls.size() << "]\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
