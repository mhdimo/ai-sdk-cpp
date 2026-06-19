#include <ai/agent/agent.hpp>

#include <stdexcept>

namespace ai {

Task<StreamTextResult> Agent::stream(
    std::string /*prompt*/,
    CancellationToken /*cancel*/
) {
    throw std::runtime_error("Agent::stream is not supported by this agent");
    co_return StreamTextResult{}; // unreachable; satisfies coroutine return type
}

Task<StreamTextResult> Agent::stream(
    std::vector<Message> /*messages*/,
    CancellationToken /*cancel*/
) {
    throw std::runtime_error("Agent::stream is not supported by this agent");
    co_return StreamTextResult{}; // unreachable; satisfies coroutine return type
}

} // namespace ai
