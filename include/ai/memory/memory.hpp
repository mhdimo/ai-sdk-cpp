#pragma once

#include <ai/session/context_strategy.hpp>  // ContextStrategy, Summarizer, Task, Prompt
#include <ai/session/session.hpp>           // Session (for the checkpoint hook signature)
#include <ai/model/embedding_model.hpp>
#include <ai/core/embed.hpp>
#include <ai/tool/tool.hpp>
#include <ai/tool/tool_set.hpp>

#include <boost/json.hpp>

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ai::memory {

/// A single unit of persistent memory.
struct MemoryRecord {
    std::string id;
    std::string scope;   ///< "project" | "checkpoint" | "scratch" | "task"
    std::string key;     ///< short title / label
    std::string content;
    std::vector<std::string> tags;
};

/// Persistent, queryable memory. Synchronous (local file I/O).
class MemoryStore {
public:
    virtual ~MemoryStore() = default;
    /// Add a record; returns its id (assigns one if empty).
    virtual std::string add(MemoryRecord rec) = 0;
    virtual void update(MemoryRecord rec) = 0;
    virtual void remove(const std::string& id) = 0;
    virtual std::optional<MemoryRecord> get(const std::string& id) const = 0;
    /// All records, optionally filtered by scope.
    virtual std::vector<MemoryRecord> list(const std::string& scope = {}) const = 0;
};

/// File-backed store. Each record is a human-editable markdown file under
/// `<dir>/<scope>/<id>.md` with `<!-- field: value -->` metadata comments
/// followed by free-form content.
class MarkdownMemoryStore : public MemoryStore {
public:
    explicit MarkdownMemoryStore(std::filesystem::path dir);
    std::string add(MemoryRecord rec) override;
    void update(MemoryRecord rec) override;
    void remove(const std::string& id) override;
    std::optional<MemoryRecord> get(const std::string& id) const override;
    std::vector<MemoryRecord> list(const std::string& scope = {}) const override;
    const std::filesystem::path& dir() const { return dir_; }

private:
    std::filesystem::path dir_;
    std::filesystem::path path_for(const std::string& scope, const std::string& id) const;
};

/// A memory record with a relevance score from a query.
struct ScoredMemory {
    MemoryRecord record;
    double score = 0.0;
};

/// Retrieves relevant memory for a query. Async because the embedding-backed
/// retriever calls `embed()`.
class MemoryRetriever {
public:
    virtual ~MemoryRetriever() = default;
    virtual Task<std::vector<ScoredMemory>> query(const std::string& q, int k) = 0;
};

/// Zero-dependency retriever: scores by query-term overlap with content/key/tags.
class KeywordRetriever : public MemoryRetriever {
public:
    explicit KeywordRetriever(MemoryStore& store);
    Task<std::vector<ScoredMemory>> query(const std::string& q, int k) override;

private:
    MemoryStore& store_;
};

/// Semantic retriever using the SDK's `embed()`: embeds record contents (cached,
/// refreshed when records change) and ranks by cosine similarity to the query.
/// A genuine differentiator vs plain FTS — leverages an existing SDK capability.
class EmbeddingRetriever : public MemoryRetriever {
public:
    EmbeddingRetriever(MemoryStore& store, EmbeddingModelPtr model);
    Task<std::vector<ScoredMemory>> query(const std::string& q, int k) override;

private:
    MemoryStore& store_;
    EmbeddingModelPtr model_;
    std::vector<MemoryRecord> cached_records_;
    std::vector<std::vector<float>> cached_embeddings_;
};

/// Combines a keyword and an embedding retriever, min-max normalizing each and
/// summing scores.
class HybridRetriever : public MemoryRetriever {
public:
    HybridRetriever(std::shared_ptr<MemoryRetriever> keyword,
                    std::shared_ptr<MemoryRetriever> embedding);
    Task<std::vector<ScoredMemory>> query(const std::string& q, int k) override;

private:
    std::shared_ptr<MemoryRetriever> keyword_;
    std::shared_ptr<MemoryRetriever> embedding_;
};

/// A ContextStrategy that decorates an inner window/compaction strategy. Before
/// delegating, it injects a compact "## Project memory" block: persistent
/// project-scope records, the latest checkpoint, and the top-K records relevant
/// to the latest user message — all within a token budget that counts toward
/// the context window.
class MemoryContextStrategy : public ContextStrategy {
public:
    MemoryContextStrategy(
        std::shared_ptr<MemoryStore> store,
        std::shared_ptr<MemoryRetriever> retriever,
        std::shared_ptr<ContextStrategy> inner,
        std::size_t memory_char_budget = 8 * 1024,
        int top_k = 4
    );
    Task<Prompt> manage(
        Prompt history,
        const ContextWindow& window,
        const TokenCounter& counter,
        Summarizer& summarizer
    ) override;

private:
    std::shared_ptr<MemoryStore> store_;
    std::shared_ptr<MemoryRetriever> retriever_;
    std::shared_ptr<ContextStrategy> inner_;
    std::size_t memory_char_budget_;
    int top_k_;
};

/// A ToolSet of memory-maintenance tools the agent can call:
/// recall_memory(query), save_memory(scope,key,content,tags?), add_note(text),
/// update_checkpoint(content), log_task_progress(task_id, entry).
ToolSet memory_tools(std::shared_ptr<MemoryStore> store,
                     std::shared_ptr<MemoryRetriever> retriever);

/// Build a Session on_turn_finish hook that, every `every_n_turns` turns,
/// summarizes the conversation into a checkpoint memory record (the
/// "continuously improves itself" loop). Best-effort: errors are swallowed.
std::function<Task<void>(const Session&)> make_checkpoint_writer(
    std::shared_ptr<MemoryStore> store,
    Summarizer summarizer,
    int every_n_turns = 5
);

} // namespace ai::memory
