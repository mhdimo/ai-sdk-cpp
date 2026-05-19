"""AI SDK - Native C++ AI agent framework with Python bindings."""

from ai_sdk._native import (
    Context,
    Provider,
    Model,
    ToolSet,
    Agent,
    generate_text,
    stream_text,
    version,
)

__all__ = [
    "Context",
    "Provider",
    "Model",
    "ToolSet",
    "Agent",
    "generate_text",
    "stream_text",
    "create_anthropic",
    "create_openai",
    "create_google",
    "create_groq",
    "create_xai",
    "create_mistral",
    "create_fireworks",
    "create_togetherai",
    "create_perplexity",
    "create_cohere",
    "create_deepseek",
    "tool",
    "version",
]

# Global context (created once, lazily)
_ctx = None

def _get_ctx():
    global _ctx
    if _ctx is None:
        _ctx = Context()
    return _ctx

def create_anthropic(api_key=None, base_url=None):
    return Provider(_get_ctx(), "anthropic", api_key=api_key, base_url=base_url)

def create_openai(api_key=None, base_url=None):
    return Provider(_get_ctx(), "openai", api_key=api_key, base_url=base_url)

def create_google(api_key=None, base_url=None):
    return Provider(_get_ctx(), "google", api_key=api_key, base_url=base_url)

def create_groq(api_key=None, base_url=None):
    return Provider(_get_ctx(), "groq", api_key=api_key, base_url=base_url)

def create_xai(api_key=None, base_url=None):
    return Provider(_get_ctx(), "xai", api_key=api_key, base_url=base_url)

def create_mistral(api_key=None, base_url=None):
    return Provider(_get_ctx(), "mistral", api_key=api_key, base_url=base_url)

def create_fireworks(api_key=None, base_url=None):
    return Provider(_get_ctx(), "fireworks", api_key=api_key, base_url=base_url)

def create_togetherai(api_key=None, base_url=None):
    return Provider(_get_ctx(), "togetherai", api_key=api_key, base_url=base_url)

def create_perplexity(api_key=None, base_url=None):
    return Provider(_get_ctx(), "perplexity", api_key=api_key, base_url=base_url)

def create_cohere(api_key=None, base_url=None):
    return Provider(_get_ctx(), "cohere", api_key=api_key, base_url=base_url)

def create_deepseek(api_key=None, base_url=None):
    return Provider(_get_ctx(), "deepseek", api_key=api_key, base_url=base_url)

def tool(name, schema, description=""):
    """Decorator to register a function as a tool."""
    def decorator(fn):
        fn._tool_name = name
        fn._tool_schema = schema
        fn._tool_description = description
        return fn
    return decorator
