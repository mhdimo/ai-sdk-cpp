#include <ai/memory/memory.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace ai::memory {

namespace {

// --- small text / id helpers ----------------------------------------------

std::string sanitize_id(const std::string& id) {
    std::string out;
    out.reserve(id.size());
    for (char c : id) {
        out += (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') ? c : '_';
    }
    return out;
}

std::string new_id() {
    std::random_device rd;
    std::uniform_int_distribution<unsigned> dist(0, 0xFFFFFFFFu);
    std::ostringstream ss;
    ss << "mem-" << std::hex << dist(rd) << dist(rd);
    return ss.str();
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

std::vector<std::string> tokenize(std::string s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            cur += std::tolower(c);
        } else {
            if (!cur.empty()) {
                out.push_back(std::move(cur));
                cur.clear();
            }
        }
    }
    if (!cur.empty()) out.push_back(std::move(cur));
    return out;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

std::string excerpt(std::string s, std::size_t max_chars) {
    if (s.size() <= max_chars) return s;
    return s.substr(0, max_chars) + "...";
}

std::string tags_to_line(const std::vector<std::string>& tags) {
    std::string out;
    for (std::size_t i = 0; i < tags.size(); ++i) {
        if (i) out += ", ";
        out += tags[i];
    }
    return out;
}

// --- markdown record file IO ----------------------------------------------

void write_record(const std::filesystem::path& path, const MemoryRecord& rec) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot write memory: " + path.string());
    f << "<!-- id: " << rec.id << " -->\n";
    f << "<!-- scope: " << rec.scope << " -->\n";
    f << "<!-- key: " << rec.key << " -->\n";
    if (!rec.tags.empty()) f << "<!-- tags: " << tags_to_line(rec.tags) << " -->\n";
    f << "\n" << rec.content;
}

std::optional<MemoryRecord> read_record(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::nullopt;
    std::vector<std::string> lines;
    {
        std::string line;
        while (std::getline(f, line)) lines.push_back(std::move(line));
    }
    if (lines.empty()) return std::nullopt;

    MemoryRecord rec;
    const std::string prefix = "<!-- ";
    const std::string mid = ": ";
    const std::string suffix = " -->";
    std::size_t i = 0;
    for (; i < lines.size(); ++i) {
        const std::string& l = lines[i];
        if (l.rfind(prefix, 0) != 0 || l.size() < suffix.size() ||
            l.compare(l.size() - suffix.size(), suffix.size(), suffix) != 0) {
            break;  // content begins here
        }
        std::string body = l.substr(prefix.size(), l.size() - prefix.size() - suffix.size());
        auto pos = body.find(mid);
        if (pos == std::string::npos) break;
        std::string key = body.substr(0, pos);
        std::string val = body.substr(pos + mid.size());
        if (key == "id") rec.id = val;
        else if (key == "scope") rec.scope = val;
        else if (key == "key") rec.key = val;
        else if (key == "tags") {
            for (auto& tok : tokenize(val)) rec.tags.push_back(tok);
        }
    }
    if (rec.id.empty()) {
        // Not a record file (no id metadata); ignore.
        return std::nullopt;
    }
    // Skip the single blank separator line written between metadata and content.
    if (i < lines.size() && lines[i].empty()) ++i;
    // Reconstruct content (remaining lines, joined).
    std::string content;
    for (; i < lines.size(); ++i) {
        content += lines[i];
        content += '\n';
    }
    while (!content.empty() && content.back() == '\n') content.pop_back();
    rec.content = content;
    return rec;
}

std::vector<std::string> parse_tags(const boost::json::value& input, const std::string& key) {
    std::vector<std::string> tags;
    const auto* obj = input.if_object();
    if (!obj) return tags;
    auto it = obj->find(key);
    if (it == obj->end()) return tags;
    if (it->value().is_array()) {
        for (auto& t : it->value().as_array()) {
            if (t.is_string()) tags.push_back(std::string(t.as_string()));
        }
    } else if (it->value().is_string()) {
        for (auto& t : tokenize(std::string(it->value().as_string()))) tags.push_back(t);
    }
    return tags;
}

std::string get_str(const boost::json::value& input, const std::string& key) {
    const auto* obj = input.if_object();
    if (!obj) return {};
    auto it = obj->find(key);
    if (it == obj->end() || !it->value().is_string()) return {};
    return std::string(it->value().as_string());
}

} // namespace

// ---------------------------------------------------------------------------
// MarkdownMemoryStore
// ---------------------------------------------------------------------------

MarkdownMemoryStore::MarkdownMemoryStore(std::filesystem::path dir)
    : dir_(std::move(dir)) {}

std::filesystem::path MarkdownMemoryStore::path_for(
    const std::string& scope, const std::string& id) const {
    std::string sid = sanitize_id(id);
    if (sid.empty()) sid = "record";
    std::string sscope = sanitize_id(scope);
    if (sscope.empty()) sscope = "misc";
    return dir_ / sscope / (sid + ".md");
}

std::string MarkdownMemoryStore::add(MemoryRecord rec) {
    if (rec.id.empty()) rec.id = new_id();
    write_record(path_for(rec.scope, rec.id), rec);
    return rec.id;
}

void MarkdownMemoryStore::update(MemoryRecord rec) {
    if (rec.id.empty()) throw std::runtime_error("update: record id required");
    write_record(path_for(rec.scope, rec.id), rec);
}

std::optional<MemoryRecord> MarkdownMemoryStore::get(const std::string& id) const {
    std::error_code ec;
    for (auto& entry : std::filesystem::recursive_directory_iterator(dir_, ec)) {
        if (ec || !entry.is_regular_file() || entry.path().extension() != ".md") {
            ec.clear();
            continue;
        }
        if (entry.path().stem().string() == sanitize_id(id)) {
            return read_record(entry.path());
        }
    }
    return std::nullopt;
}

void MarkdownMemoryStore::remove(const std::string& id) {
    std::error_code ec;
    for (auto& entry : std::filesystem::recursive_directory_iterator(dir_, ec)) {
        if (ec || !entry.is_regular_file() || entry.path().extension() != ".md") {
            ec.clear();
            continue;
        }
        if (entry.path().stem().string() == sanitize_id(id)) {
            std::filesystem::remove(entry.path(), ec);
            return;
        }
    }
}

std::vector<MemoryRecord> MarkdownMemoryStore::list(const std::string& scope) const {
    std::vector<MemoryRecord> out;
    std::error_code ec;
    std::filesystem::path root = scope.empty() ? dir_ : (dir_ / sanitize_id(scope));
    if (!std::filesystem::exists(root)) return out;
    for (auto& entry : std::filesystem::recursive_directory_iterator(root, ec)) {
        if (ec || !entry.is_regular_file() || entry.path().extension() != ".md") {
            ec.clear();
            continue;
        }
        auto rec = read_record(entry.path());
        if (rec) out.push_back(std::move(*rec));
    }
    return out;
}

// ---------------------------------------------------------------------------
// KeywordRetriever
// ---------------------------------------------------------------------------

KeywordRetriever::KeywordRetriever(MemoryStore& store) : store_(store) {}

Task<std::vector<ScoredMemory>> KeywordRetriever::query(const std::string& q, int k) {
    auto terms = tokenize(q);
    std::vector<ScoredMemory> scored;
    for (auto& rec : store_.list()) {
        std::string hay = to_lower(rec.key + " " + rec.content + " " + tags_to_line(rec.tags));
        double score = 0.0;
        for (auto& t : terms) {
            std::string::size_type pos = 0;
            while ((pos = hay.find(t, pos)) != std::string::npos) {
                score += 1.0;
                pos += t.size();
            }
        }
        if (score > 0.0) scored.push_back({rec, score});
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    if ((int)scored.size() > k) scored.resize(k);
    co_return scored;
}

// ---------------------------------------------------------------------------
// EmbeddingRetriever
// ---------------------------------------------------------------------------

EmbeddingRetriever::EmbeddingRetriever(MemoryStore& store, EmbeddingModelPtr model)
    : store_(store), model_(std::move(model)) {}

Task<std::vector<ScoredMemory>> EmbeddingRetriever::query(const std::string& q, int k) {
    if (!model_) co_return std::vector<ScoredMemory>{};

    auto records = store_.list();

    // Refresh the embedding cache when the record set changes.
    auto same_set = [&]() {
        if (records.size() != cached_records_.size()) return false;
        for (std::size_t i = 0; i < records.size(); ++i) {
            if (records[i].id != cached_records_[i].id) return false;
        }
        return true;
    };
    if (!same_set()) {
        std::vector<std::string> values;
        values.reserve(records.size());
        for (auto& r : records) {
            values.push_back(r.key + "\n" + r.content);
        }
        EmbedManyOptions opts;
        opts.model = model_;
        opts.values = std::move(values);
        auto result = co_await embed_many(std::move(opts));
        cached_records_ = records;
        cached_embeddings_ = std::move(result.embeddings);
    }

    SingleEmbedOptions qopts;
    qopts.model = model_;
    qopts.value = q;
    auto qres = co_await embed(std::move(qopts));

    std::vector<ScoredMemory> scored;
    for (std::size_t i = 0; i < cached_records_.size() && i < cached_embeddings_.size(); ++i) {
        scored.push_back({cached_records_[i], cosine(qres.embedding, cached_embeddings_[i])});
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    if ((int)scored.size() > k) scored.resize(k);
    co_return scored;
}

// ---------------------------------------------------------------------------
// HybridRetriever
// ---------------------------------------------------------------------------

HybridRetriever::HybridRetriever(std::shared_ptr<MemoryRetriever> keyword,
                                 std::shared_ptr<MemoryRetriever> embedding)
    : keyword_(std::move(keyword)), embedding_(std::move(embedding)) {}

Task<std::vector<ScoredMemory>> HybridRetriever::query(const std::string& q, int k) {
    auto kw = keyword_ ? co_await keyword_->query(q, k * 2) : std::vector<ScoredMemory>{};
    auto em = embedding_ ? co_await embedding_->query(q, k * 2) : std::vector<ScoredMemory>{};

    auto normalize = [](std::vector<ScoredMemory>& v) {
        if (v.empty()) return;
        double mx = v.front().score, mn = v.back().score;
        for (auto& s : v) mx = std::max(mx, s.score), mn = std::min(mn, s.score);
        double range = mx - mn;
        if (range <= 0.0) {
            for (auto& s : v) s.score = 1.0;
        } else {
            for (auto& s : v) s.score = (s.score - mn) / range;
        }
    };
    normalize(kw);
    normalize(em);

    // Merge by record id, summing normalized scores.
    std::unordered_map<std::string, ScoredMemory> merged;
    auto fold = [&](std::vector<ScoredMemory>& v) {
        for (auto& s : v) {
            auto it = merged.find(s.record.id);
            if (it == merged.end()) {
                merged[s.record.id] = s;
            } else {
                it->second.score += s.score;
            }
        }
    };
    fold(kw);
    fold(em);

    std::vector<ScoredMemory> out;
    out.reserve(merged.size());
    for (auto& [_, s] : merged) out.push_back(std::move(s));
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    if ((int)out.size() > k) out.resize(k);
    co_return out;
}

// ---------------------------------------------------------------------------
// MemoryContextStrategy
// ---------------------------------------------------------------------------

MemoryContextStrategy::MemoryContextStrategy(
    std::shared_ptr<MemoryStore> store,
    std::shared_ptr<MemoryRetriever> retriever,
    std::shared_ptr<ContextStrategy> inner,
    std::size_t memory_char_budget,
    int top_k)
    : store_(std::move(store)),
      retriever_(std::move(retriever)),
      inner_(std::move(inner)),
      memory_char_budget_(memory_char_budget),
      top_k_(top_k) {}

namespace {
std::string latest_user_text(const Prompt& history) {
    for (auto it = history.rbegin(); it != history.rend(); ++it) {
        if (auto* u = std::get_if<UserMessage>(&*it)) {
            std::string text;
            for (auto& p : u->content) {
                if (auto* t = std::get_if<TextPart>(&p)) text += t->text;
            }
            return text;
        }
    }
    return {};
}

void append_capped(std::string& block, const std::string& line, std::size_t budget) {
    if (block.size() + line.size() + 1 > budget) return;
    block += line;
    block += '\n';
}
} // namespace

Task<Prompt> MemoryContextStrategy::manage(
    Prompt history,
    const ContextWindow& window,
    const TokenCounter& counter,
    Summarizer& summarizer
) {
    if (store_) {
        std::unordered_set<std::string> seen;
        std::string block;

        // Persistent project knowledge.
        for (auto& rec : store_->list("project")) {
            if (seen.insert(rec.id).second) {
                append_capped(block, "- **" + rec.key + "**: " + excerpt(rec.content, 500),
                              memory_char_budget_);
            }
        }
        // Latest checkpoint.
        auto checkpoints = store_->list("checkpoint");
        if (!checkpoints.empty()) {
            auto& cp = checkpoints.back();
            if (seen.insert(cp.id).second) {
                append_capped(block, "\n### Checkpoint\n" + excerpt(cp.content, 1000),
                              memory_char_budget_);
            }
        }
        // Relevant memories for the latest user message.
        std::string q = latest_user_text(history);
        if (retriever_ && !q.empty()) {
            auto hits = co_await retriever_->query(q, top_k_);
            for (auto& h : hits) {
                if (seen.insert(h.record.id).second) {
                    append_capped(block,
                                  "- **" + h.record.key + "** (" + h.record.scope + "): " +
                                      excerpt(h.record.content, 400),
                                  memory_char_budget_);
                }
            }
        }

        if (!block.empty()) {
            std::string memory_text = "## Project memory\n" + block;
            // Insert after leading system message(s).
            std::size_t insert_at = 0;
            while (insert_at < history.size() &&
                   std::holds_alternative<SystemMessage>(history[insert_at])) {
                ++insert_at;
            }
            history.insert(history.begin() + insert_at,
                           SystemMessage{.content = std::move(memory_text)});
        }
    }

    // Delegate window/compaction to the inner strategy; memory counts toward
    // the budget because the inner strategy respects the ContextWindow.
    if (inner_) {
        co_return co_await inner_->manage(std::move(history), window, counter, summarizer);
    }
    co_return history;
}

// ---------------------------------------------------------------------------
// memory_tools
// ---------------------------------------------------------------------------

ToolSet memory_tools(std::shared_ptr<MemoryStore> store,
                     std::shared_ptr<MemoryRetriever> retriever) {
    ToolSet tools;

    tools.add(tool(
        "recall_memory",
        schema::JsonSchema::object({{"query", schema::JsonSchema::string("What to recall")}})
            .required({"query"}),
        "Retrieve relevant memories for a query.",
        [retriever](const boost::json::value& input, ToolExecutionContext)
            -> Task<boost::json::value> {
            if (!retriever) co_return boost::json::array{};
            std::string q = get_str(input, "query");
            auto hits = co_await retriever->query(q, 5);
            boost::json::array out;
            for (auto& h : hits) {
                out.push_back(boost::json::object{
                    {"key", h.record.key},
                    {"scope", h.record.scope},
                    {"content", excerpt(h.record.content, 500)},
                    {"score", h.score},
                });
            }
            co_return boost::json::value(out);
        }));

    tools.add(tool(
        "save_memory",
        schema::JsonSchema::object({
            {"scope", schema::JsonSchema::string("project | checkpoint | scratch | task")},
            {"key", schema::JsonSchema::string("Short title")},
            {"content", schema::JsonSchema::string("Memory content")},
            {"tags", schema::JsonSchema::array(schema::JsonSchema::string())},
        }).required({"scope", "key", "content"}),
        "Persist a memory record across sessions.",
        [store](const boost::json::value& input, ToolExecutionContext)
            -> Task<boost::json::value> {
            if (!store) co_return boost::json::value("no memory store");
            MemoryRecord rec;
            rec.scope = get_str(input, "scope");
            rec.key = get_str(input, "key");
            rec.content = get_str(input, "content");
            rec.tags = parse_tags(input, "tags");
            std::string id = store->add(std::move(rec));
            co_return boost::json::value("saved memory " + id);
        }));

    tools.add(tool(
        "add_note",
        schema::JsonSchema::object({{"content", schema::JsonSchema::string("Note text")}})
            .required({"content"}),
        "Add a temporary scratch note.",
        [store](const boost::json::value& input, ToolExecutionContext)
            -> Task<boost::json::value> {
            if (!store) co_return boost::json::value("no memory store");
            MemoryRecord rec;
            rec.scope = "scratch";
            rec.key = "note";
            rec.content = get_str(input, "content");
            std::string id = store->add(std::move(rec));
            co_return boost::json::value("saved note " + id);
        }));

    tools.add(tool(
        "update_checkpoint",
        schema::JsonSchema::object({{"content", schema::JsonSchema::string("Checkpoint state")}})
            .required({"content"}),
        "Overwrite the session checkpoint memory.",
        [store](const boost::json::value& input, ToolExecutionContext)
            -> Task<boost::json::value> {
            if (!store) co_return boost::json::value("no memory store");
            std::string content = get_str(input, "content");
            auto existing = store->list("checkpoint");
            MemoryRecord rec;
            rec.scope = "checkpoint";
            rec.key = "checkpoint";
            rec.content = std::move(content);
            if (!existing.empty()) {
                rec.id = existing.front().id;
                store->update(rec);
            } else {
                rec.id = store->add(rec);
            }
            co_return boost::json::value("checkpoint updated");
        }));

    tools.add(tool(
        "log_task_progress",
        schema::JsonSchema::object({
            {"task_id", schema::JsonSchema::string("Task identifier")},
            {"entry", schema::JsonSchema::string("Progress entry to append")},
        }).required({"task_id", "entry"}),
        "Append a progress entry to a task's log.",
        [store](const boost::json::value& input, ToolExecutionContext)
            -> Task<boost::json::value> {
            if (!store) co_return boost::json::value("no memory store");
            std::string task_id = get_str(input, "task_id");
            std::string entry = get_str(input, "entry");
            MemoryRecord rec;
            rec.scope = "task";
            rec.key = task_id;
            rec.id = task_id;
            // Append to existing content if present.
            auto existing = store->get(task_id);
            if (existing && existing->scope == "task") {
                rec.content = existing->content + "\n- " + entry;
            } else {
                rec.content = "- " + entry;
            }
            if (existing) store->update(rec);
            else store->add(rec);
            co_return boost::json::value("logged progress for task " + task_id);
        }));

    return tools;
}

// ---------------------------------------------------------------------------
// CheckpointWriter (Session post-turn hook)
// ---------------------------------------------------------------------------

std::function<Task<void>(const Session&)> make_checkpoint_writer(
    std::shared_ptr<MemoryStore> store,
    Summarizer summarizer,
    int every_n_turns
) {
    return [store, summarizer = std::move(summarizer), every_n_turns](
               const Session& session) -> Task<void> {
        if (!store || !summarizer) co_return;
        int turns = session.metadata().turns;
        if (turns <= 0 || every_n_turns <= 0 || (turns % every_n_turns) != 0) co_return;
        try {
            std::string summary = co_await summarizer(session.history());
            if (summary.empty()) co_return;
            MemoryRecord rec;
            rec.scope = "checkpoint";
            rec.key = "checkpoint";
            rec.content = std::move(summary);
            auto existing = store->list("checkpoint");
            if (!existing.empty()) {
                rec.id = existing.front().id;
                store->update(rec);
            } else {
                store->add(rec);
            }
        } catch (...) {
            // Best-effort: never break the agent loop on a checkpoint failure.
        }
        co_return;
    };
}

} // namespace ai::memory
