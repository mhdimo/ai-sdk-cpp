#include <ai/coding_agent.hpp>

#include <ai/core/generate_text.hpp>
#include <ai/prompt/message.hpp>

#include <type_traits>
#include <utility>

namespace ai {

namespace {

// Render a conversation transcript for summarization (text view of messages).
std::string render_transcript(const Prompt& history) {
    std::string out;
    for (auto& msg : history) {
        std::visit(
            [&](auto& m) {
                using T = std::decay_t<decltype(m)>;
                if constexpr (std::is_same_v<T, SystemMessage>) {
                    out += "[system] " + m.content + "\n";
                } else if constexpr (std::is_same_v<T, UserMessage>) {
                    out += "[user] ";
                    for (auto& p : m.content) {
                        if (auto* t = std::get_if<TextPart>(&p)) out += t->text;
                    }
                    out += "\n";
                } else if constexpr (std::is_same_v<T, AssistantMessage>) {
                    out += "[assistant] ";
                    for (auto& p : m.content) {
                        if (auto* t = std::get_if<TextPart>(&p)) {
                            out += t->text;
                        } else if (auto* tc = std::get_if<ToolCallPart>(&p)) {
                            out += "(call " + tc->tool_name + ") ";
                        }
                    }
                    out += "\n";
                } else if constexpr (std::is_same_v<T, ToolMessage>) {
                    out += "[tool] ";
                    for (auto& tr : m.content) {
                        if (auto* t = std::get_if<TextOutput>(&tr.output)) out += t->value;
                    }
                    out += "\n";
                }
            },
            msg);
    }
    return out;
}

// Default checkpoint summarizer: ask the model to condense the transcript.
Summarizer default_summarizer(LanguageModelPtr model) {
    return [model](const Prompt& history) -> Task<std::string> {
        GenerateTextOptions o;
        o.model = model;
        o.system = "Summarize the conversation so far into a concise checkpoint: "
                   "key decisions, current state, and next steps.";
        o.prompt = render_transcript(history);
        auto r = co_await generate_text(std::move(o));
        co_return r.text;
    };
}

std::shared_ptr<memory::MemoryRetriever> make_retriever(
    const std::shared_ptr<memory::MemoryStore>& store,
    const EmbeddingModelPtr& embedding_model
) {
    if (!store) return nullptr;
    auto kw = std::make_shared<memory::KeywordRetriever>(*store);
    if (embedding_model) {
        auto em = std::make_shared<memory::EmbeddingRetriever>(*store, embedding_model);
        return std::make_shared<memory::HybridRetriever>(kw, em);
    }
    return kw;
}

ToolLoopAgentOptions build_agent_options(
    const CodingAgentOptions& opts,
    const std::shared_ptr<memory::MemoryStore>& store,
    const std::shared_ptr<memory::MemoryRetriever>& retriever
) {
    ToolSet tools = standard_toolkit(opts.toolkit_options);
    if (store && retriever) {
        ToolSet mt = memory::memory_tools(store, retriever);
        for (auto& [name, tool] : mt) {
            tools.add(tool);
        }
    }
    if (opts.permission_policy || opts.approver) {
        tools = with_permissions(std::move(tools), opts.permission_policy, opts.approver);
    }

    ToolLoopAgentOptions ao;
    ao.model = opts.model;
    ao.tools = std::move(tools);
    ao.instructions = opts.instructions;
    ao.max_steps = opts.max_steps;
    return ao;
}

Session build_session(
    Agent& agent,
    const CodingAgentOptions& opts,
    const std::shared_ptr<memory::MemoryStore>& store,
    const std::shared_ptr<memory::MemoryRetriever>& retriever
) {
    std::shared_ptr<ContextStrategy> strategy;
    if (store && retriever) {
        strategy = std::make_shared<memory::MemoryContextStrategy>(
            store, retriever, std::make_shared<SlidingWindowStrategy>());
    } else {
        strategy = std::make_shared<SlidingWindowStrategy>();
    }

    Session s(agent, opts.window, std::make_shared<ApproximateTokenCounter>(), strategy, {});

    if (store) {
        Summarizer summarizer = opts.checkpoint_summarizer
                                    ? opts.checkpoint_summarizer
                                    : default_summarizer(opts.model);
        s.set_on_turn_finish(
            memory::make_checkpoint_writer(store, summarizer, opts.checkpoint_every_n_turns));
    }
    return s;
}

} // namespace

CodingAgent::CodingAgent(CodingAgentOptions opts)
    : model_(opts.model),
      store_(opts.enable_memory
                 ? std::make_shared<memory::MarkdownMemoryStore>(opts.memory_dir)
                 : nullptr),
      retriever_(make_retriever(store_, opts.embedding_model)),
      agent_(build_agent_options(opts, store_, retriever_)),
      session_(build_session(agent_, opts, store_, retriever_)) {}

Task<GenerateTextResult> CodingAgent::send(std::string user_msg, CancellationToken cancel) {
    co_return co_await session_.send(std::move(user_msg), std::move(cancel));
}

} // namespace ai
