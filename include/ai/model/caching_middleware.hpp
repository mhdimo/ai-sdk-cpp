#pragma once

#include <ai/model/middleware.hpp>
#include <ai/util/json.hpp>
#include <boost/json.hpp>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ai {

/// A middleware that provides simple in-memory LRU caching for identical prompts.
/// Only caches generate calls (not streaming).
class CachingMiddleware : public Middleware {
public:
    /// Constructs a caching middleware with the given max number of entries.
    explicit CachingMiddleware(size_t max_entries = 100)
        : max_entries_(max_entries) {}

    Task<GenerateResult> wrap_generate(
        GenerateFn do_generate,
        StreamFn do_stream,
        CallOptions& options
    ) override {
        auto key = compute_cache_key(options);

        {
            std::lock_guard lock(mutex_);
            auto it = cache_map_.find(key);
            if (it != cache_map_.end()) {
                // Move to front (most recently used)
                cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
                co_return it->second->second;
            }
        }

        auto result = co_await do_generate();

        {
            std::lock_guard lock(mutex_);
            // Evict LRU if at capacity
            while (cache_list_.size() >= max_entries_) {
                auto last = cache_list_.end();
                --last;
                cache_map_.erase(last->first);
                cache_list_.erase(last);
            }
            // Insert new entry at front
            cache_list_.emplace_front(key, result);
            cache_map_[key] = cache_list_.begin();
        }

        co_return result;
    }

    Task<StreamResult> wrap_stream(
        GenerateFn do_generate,
        StreamFn do_stream,
        CallOptions& options
    ) override {
        // Streaming is not cached; pass through directly
        co_return co_await do_stream();
    }

    /// Clears the entire cache.
    void clear() {
        std::lock_guard lock(mutex_);
        cache_list_.clear();
        cache_map_.clear();
    }

    /// Returns the current number of cached entries.
    size_t size() const {
        std::lock_guard lock(mutex_);
        return cache_list_.size();
    }

private:
    size_t max_entries_;
    mutable std::mutex mutex_;

    using CacheEntry = std::pair<std::string, GenerateResult>;
    std::list<CacheEntry> cache_list_;
    std::unordered_map<std::string, std::list<CacheEntry>::iterator> cache_map_;

    static std::string compute_cache_key(const CallOptions& options) {
        // Build a JSON representation of the prompt + key parameters for hashing.
        boost::json::object key_obj;

        // Serialize the prompt messages as JSON array
        boost::json::array messages;
        for (auto& msg : options.prompt) {
            if (auto* sys = std::get_if<SystemMessage>(&msg)) {
                boost::json::object m;
                m["role"] = "system";
                m["content"] = sys->content;
                messages.push_back(std::move(m));
            } else if (auto* user = std::get_if<UserMessage>(&msg)) {
                boost::json::object m;
                m["role"] = "user";
                boost::json::array parts;
                for (auto& part : user->content) {
                    if (auto* tp = std::get_if<TextPart>(&part)) {
                        boost::json::object p;
                        p["type"] = "text";
                        p["text"] = tp->text;
                        parts.push_back(std::move(p));
                    }
                }
                m["content"] = std::move(parts);
                messages.push_back(std::move(m));
            } else if (auto* assistant = std::get_if<AssistantMessage>(&msg)) {
                boost::json::object m;
                m["role"] = "assistant";
                boost::json::array parts;
                for (auto& part : assistant->content) {
                    if (auto* tp = std::get_if<TextPart>(&part)) {
                        boost::json::object p;
                        p["type"] = "text";
                        p["text"] = tp->text;
                        parts.push_back(std::move(p));
                    } else if (auto* tcp = std::get_if<ToolCallPart>(&part)) {
                        boost::json::object p;
                        p["type"] = "tool_call";
                        p["name"] = tcp->tool_name;
                        p["input"] = tcp->input;
                        parts.push_back(std::move(p));
                    }
                }
                m["content"] = std::move(parts);
                messages.push_back(std::move(m));
            } else if (auto* tool_msg = std::get_if<ToolMessage>(&msg)) {
                boost::json::object m;
                m["role"] = "tool";
                m["id"] = tool_msg->content.empty() ? "" : tool_msg->content[0].tool_call_id;
                messages.push_back(std::move(m));
            }
        }
        key_obj["messages"] = std::move(messages);

        if (options.max_output_tokens)
            key_obj["max_output_tokens"] = *options.max_output_tokens;
        if (options.temperature)
            key_obj["temperature"] = *options.temperature;
        if (options.top_p)
            key_obj["top_p"] = *options.top_p;
        if (options.seed)
            key_obj["seed"] = *options.seed;

        // Serialize tool names
        if (!options.tools.empty()) {
            boost::json::array tool_names;
            for (auto& t : options.tools) {
                tool_names.push_back(boost::json::value(t.name));
            }
            key_obj["tools"] = std::move(tool_names);
        }

        return boost::json::serialize(key_obj);
    }
};

inline MiddlewarePtr make_caching_middleware(size_t max_entries = 100) {
    return std::make_shared<CachingMiddleware>(max_entries);
}

} // namespace ai
