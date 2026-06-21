#include <ai/session/store.hpp>

#include <boost/json.hpp>

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace ai {

namespace {

// --- string/atom helpers (defensive, never throw on missing fields) --------

std::string string_or(const boost::json::object& obj, const char* key,
                      const std::string& fallback) {
    auto it = obj.find(key);
    if (it != obj.end() && it->value().is_string()) {
        return std::string(it->value().as_string());
    }
    return fallback;
}

int int_or(const boost::json::object& obj, const char* key, int fallback) {
    auto it = obj.find(key);
    if (it != obj.end() && it->value().is_int64()) {
        return static_cast<int>(it->value().as_int64());
    }
    return fallback;
}

std::string str_at(const boost::json::object& obj, const char* key) {
    return std::string(obj.at(key).as_string());
}

// --- tool result output serialization (all variants) ----------------------

boost::json::value serialize_output(const ToolResultOutput& out) {
    return std::visit(
        [](const auto& o) -> boost::json::value {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, TextOutput>) {
                return boost::json::object{{"kind", "text"}, {"value", o.value}};
            } else if constexpr (std::is_same_v<T, JsonOutput>) {
                return boost::json::object{{"kind", "json"}, {"value", o.value}};
            } else if constexpr (std::is_same_v<T, ErrorTextOutput>) {
                return boost::json::object{{"kind", "error_text"}, {"value", o.value}};
            } else if constexpr (std::is_same_v<T, ErrorJsonOutput>) {
                return boost::json::object{{"kind", "error_json"}, {"value", o.value}};
            } else if constexpr (std::is_same_v<T, ExecutionDenied>) {
                boost::json::object obj{{"kind", "denied"}};
                if (o.reason) obj["reason"] = *o.reason;
                return obj;
            } else {
                // ContentOutput (structured file output) not persisted in v1.
                return boost::json::object{{"kind", "content"}};
            }
        },
        out);
}

ToolResultOutput deserialize_output(const boost::json::value& v) {
    auto& o = v.as_object();
    std::string kind = str_at(o, "kind");
    if (kind == "text") return TextOutput{.value = str_at(o, "value")};
    if (kind == "json") return JsonOutput{.value = o.at("value")};
    if (kind == "error_text") return ErrorTextOutput{.value = str_at(o, "value")};
    if (kind == "error_json") return ErrorJsonOutput{.value = o.at("value")};
    if (kind == "denied") {
        ExecutionDenied d;
        if (auto it = o.find("reason"); it != o.end() && it->value().is_string()) {
            d.reason = std::string(it->value().as_string());
        }
        return d;
    }
    return TextOutput{.value = "[unsupported tool output]"};
}

// --- message serialization (full fidelity for agent resume) ----------------

boost::json::value serialize_message(const Message& msg) {
    return std::visit(
        [](const auto& m) -> boost::json::value {
            using T = std::decay_t<decltype(m)>;
            if constexpr (std::is_same_v<T, SystemMessage>) {
                return boost::json::object{{"role", "system"}, {"content", m.content}};
            } else if constexpr (std::is_same_v<T, UserMessage>) {
                boost::json::array parts;
                for (auto& p : m.content) {
                    if (auto* t = std::get_if<TextPart>(&p)) {
                        parts.push_back(boost::json::object{{"type", "text"}, {"text", t->text}});
                    } else if (auto* f = std::get_if<FilePart>(&p)) {
                        if (auto* url = std::get_if<UrlFileData>(&f->data)) {
                            parts.push_back(boost::json::object{
                                {"type", "file_url"},
                                {"url", url->url},
                                {"media_type", f->media_type}});
                        }
                        // DataFileData/ReferenceFileData/TextFileData skipped in v1.
                    }
                }
                return boost::json::object{{"role", "user"}, {"parts", std::move(parts)}};
            } else if constexpr (std::is_same_v<T, AssistantMessage>) {
                boost::json::array parts;
                for (auto& p : m.content) {
                    if (auto* t = std::get_if<TextPart>(&p)) {
                        parts.push_back(boost::json::object{{"type", "text"}, {"text", t->text}});
                    } else if (auto* r = std::get_if<ReasoningPart>(&p)) {
                        boost::json::object rp{{"type", "reasoning"}, {"text", r->text}};
                        if (r->signature) rp["signature"] = *r->signature;
                        if (r->redacted_data) rp["redacted_data"] = *r->redacted_data;
                        parts.push_back(std::move(rp));
                    } else if (auto* tc = std::get_if<ToolCallPart>(&p)) {
                        parts.push_back(boost::json::object{
                            {"type", "tool_call"},
                            {"id", tc->tool_call_id},
                            {"name", tc->tool_name},
                            {"input", tc->input}});
                    }
                    // FilePart / ToolResultPart within an assistant message: skipped in v1.
                }
                return boost::json::object{{"role", "assistant"}, {"parts", std::move(parts)}};
            } else {  // ToolMessage
                boost::json::array parts;
                for (auto& tr : m.content) {
                    parts.push_back(boost::json::object{
                        {"type", "tool_result"},
                        {"id", tr.tool_call_id},
                        {"name", tr.tool_name},
                        {"output", serialize_output(tr.output)}});
                }
                return boost::json::object{{"role", "tool"}, {"parts", std::move(parts)}};
            }
        },
        msg);
}

Message deserialize_message(const boost::json::value& v) {
    auto& o = v.as_object();
    std::string role = str_at(o, "role");

    if (role == "system") {
        return SystemMessage{.content = str_at(o, "content")};
    }
    auto& parts = o.at("parts").as_array();

    if (role == "user") {
        UserContent c;
        for (auto& pv : parts) {
            auto& po = pv.as_object();
            std::string type = str_at(po, "type");
            if (type == "text") {
                c.push_back(TextPart{.text = str_at(po, "text")});
            } else if (type == "file_url") {
                FilePart fp;
                fp.data = UrlFileData{.url = str_at(po, "url")};
                fp.media_type = str_at(po, "media_type");
                c.push_back(std::move(fp));
            }
        }
        return UserMessage{.content = std::move(c)};
    }

    if (role == "assistant") {
        AssistantContent c;
        for (auto& pv : parts) {
            auto& po = pv.as_object();
            std::string type = str_at(po, "type");
            if (type == "text") {
                c.push_back(TextPart{.text = str_at(po, "text")});
            } else if (type == "reasoning") {
                ReasoningPart rp{.text = str_at(po, "text")};
                if (auto s = po.find("signature"); s != po.end() && s->value().is_string())
                    rp.signature = std::string(s->value().as_string());
                if (auto d = po.find("redacted_data"); d != po.end() && d->value().is_string())
                    rp.redacted_data = std::string(d->value().as_string());
                c.push_back(std::move(rp));
            } else if (type == "tool_call") {
                ToolCallPart tc;
                tc.tool_call_id = str_at(po, "id");
                tc.tool_name = str_at(po, "name");
                tc.input = po.at("input");
                c.push_back(std::move(tc));
            }
        }
        return AssistantMessage{.content = std::move(c)};
    }

    // tool
    ToolContent c;
    for (auto& pv : parts) {
        auto& po = pv.as_object();
        if (str_at(po, "type") == "tool_result") {
            ToolResultPart tr;
            tr.tool_call_id = str_at(po, "id");
            tr.tool_name = str_at(po, "name");
            tr.output = deserialize_output(po.at("output"));
            c.push_back(std::move(tr));
        }
    }
    return ToolMessage{.content = std::move(c)};
}

} // namespace

JsonFileSessionStore::JsonFileSessionStore(std::filesystem::path dir)
    : dir_(std::move(dir)) {}

std::filesystem::path JsonFileSessionStore::path_for(const std::string& id) const {
    // Sanitize to prevent path traversal.
    std::string safe;
    safe.reserve(id.size());
    for (char c : id) {
        safe += (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')
                    ? c
                    : '_';
    }
    return dir_ / (safe + ".json");
}

Task<void> JsonFileSessionStore::save(const SessionSnapshot& snap) {
    boost::json::object root;
    root["id"] = snap.id;
    root["model_id"] = snap.model_id;
    root["provider_id"] = snap.provider_id;

    boost::json::object meta;
    meta["turns"] = snap.metadata.turns;
    meta["total_input_tokens"] = snap.metadata.total_input_tokens;
    meta["total_output_tokens"] = snap.metadata.total_output_tokens;
    if (snap.metadata.compaction_summary) {
        meta["compaction_summary"] = *snap.metadata.compaction_summary;
    }
    root["metadata"] = std::move(meta);

    boost::json::array hist;
    for (auto& m : snap.history) hist.push_back(serialize_message(m));
    root["history"] = std::move(hist);

    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    std::ofstream f(path_for(snap.id), std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot write session: " + snap.id);
    f << boost::json::serialize(root);
    co_return;
}

Task<std::optional<SessionSnapshot>> JsonFileSessionStore::load(const std::string& id) {
    std::filesystem::path p = path_for(id);
    if (!std::filesystem::exists(p)) co_return std::nullopt;

    std::ifstream f(p, std::ios::binary);
    if (!f) co_return std::nullopt;
    std::stringstream ss;
    ss << f.rdbuf();

    boost::system::error_code ec;
    auto root = boost::json::parse(ss.str(), ec);
    if (ec || !root.is_object()) co_return std::nullopt;
    auto& obj = root.as_object();

    SessionSnapshot snap;
    snap.id = string_or(obj, "id", id);
    snap.model_id = string_or(obj, "model_id", {});
    snap.provider_id = string_or(obj, "provider_id", {});

    if (auto it = obj.find("metadata"); it != obj.end() && it->value().is_object()) {
        auto& m = it->value().as_object();
        snap.metadata.turns = int_or(m, "turns", 0);
        snap.metadata.total_input_tokens = int_or(m, "total_input_tokens", 0);
        snap.metadata.total_output_tokens = int_or(m, "total_output_tokens", 0);
        if (auto c = m.find("compaction_summary");
            c != m.end() && c->value().is_string()) {
            snap.metadata.compaction_summary = std::string(c->value().as_string());
        }
    }

    if (auto it = obj.find("history"); it != obj.end() && it->value().is_array()) {
        for (auto& mv : it->value().as_array()) {
            try {
                snap.history.push_back(deserialize_message(mv));
            } catch (...) {
                // Skip a malformed message rather than failing the whole load.
            }
        }
    }

    co_return snap;
}

Task<std::vector<SessionMeta>> JsonFileSessionStore::list() {
    std::vector<SessionMeta> out;
    std::error_code ec;
    for (auto it = std::filesystem::directory_iterator(dir_, ec);
         !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
        if (ec || !it->is_regular_file() || it->path().extension() != ".json") {
            ec.clear();
            continue;
        }
        std::ifstream f(it->path(), std::ios::binary);
        if (!f) continue;
        std::stringstream ss;
        ss << f.rdbuf();
        boost::system::error_code pec;
        auto root = boost::json::parse(ss.str(), pec);
        if (pec || !root.is_object()) continue;
        auto& obj = root.as_object();

        SessionMeta meta;
        meta.id = string_or(obj, "id", it->path().stem().string());
        meta.model_id = string_or(obj, "model_id", {});
        meta.provider_id = string_or(obj, "provider_id", {});
        if (auto m = obj.find("metadata"); m != obj.end() && m->value().is_object()) {
            meta.turns = int_or(m->value().as_object(), "turns", 0);
        }
        out.push_back(std::move(meta));
    }
    co_return out;
}

Task<void> JsonFileSessionStore::remove(const std::string& id) {
    std::error_code ec;
    std::filesystem::remove(path_for(id), ec);
    co_return;
}

} // namespace ai
