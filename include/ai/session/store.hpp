#pragma once

#include <ai/session/session.hpp>      // SessionSnapshot, SessionMetadata, Prompt
#include <ai/stream/async_generator.hpp>  // Task

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ai {

/// Lightweight summary of a stored session (no history) for listing.
struct SessionMeta {
    std::string id;
    int turns = 0;
    std::string model_id;
    std::string provider_id;
};

/// Persists and restores SessionSnapshots so conversations can resume across
/// runs. Implementations are coroutines so a future async-backed store can do
/// I/O without changing the interface.
class SessionStore {
public:
    virtual ~SessionStore() = default;
    virtual Task<void> save(const SessionSnapshot&) = 0;
    virtual Task<std::optional<SessionSnapshot>> load(const std::string& id) = 0;
    virtual Task<std::vector<SessionMeta>> list() = 0;
    virtual Task<void> remove(const std::string& id) = 0;
};

/// Stores each session as `<dir>/<id>.json`. Full message fidelity (system,
/// user text/file-url, assistant text/tool-call/reasoning, tool results) so
/// tool conversations resume correctly. Inline file DATA parts (images/binary)
/// are not persisted in v1 (rare in transcripts); URL file parts are.
class JsonFileSessionStore : public SessionStore {
public:
    explicit JsonFileSessionStore(std::filesystem::path dir);

    Task<void> save(const SessionSnapshot&) override;
    Task<std::optional<SessionSnapshot>> load(const std::string& id) override;
    Task<std::vector<SessionMeta>> list() override;
    Task<void> remove(const std::string& id) override;

    const std::filesystem::path& dir() const { return dir_; }

private:
    std::filesystem::path path_for(const std::string& id) const;
    std::filesystem::path dir_;
};

} // namespace ai
