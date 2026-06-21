"""AI SDK - Native C++ AI agent framework with Python bindings."""

from ai_sdk._native import (
    Context,
    Provider,
    Model,
    ToolSet,
    Agent,
    Session,
    generate_text,
    stream_text,
    standard_toolkit,
    with_permissions,
    version,
)

__all__ = [
    "Context",
    "Provider",
    "Model",
    "ToolSet",
    "Agent",
    "Session",
    "generate_text",
    "stream_text",
    "standard_toolkit",
    "with_permissions",
    "create_anthropic",
    "create_openai",
    "create_google",
    "create_deepseek",
    "create_zai",
    "create_deepseek_anthropic",
    "create_zai_openai",
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

def create_deepseek(api_key=None, base_url=None):
    return Provider(_get_ctx(), "deepseek", api_key=api_key, base_url=base_url)

def create_zai(api_key=None, base_url=None):
    return Provider(_get_ctx(), "zai", api_key=api_key, base_url=base_url)

def create_deepseek_anthropic(api_key=None, base_url=None):
    return Provider(_get_ctx(), "deepseek-anthropic", api_key=api_key, base_url=base_url)

def create_zai_openai(api_key=None, base_url=None):
    return Provider(_get_ctx(), "zai-openai", api_key=api_key, base_url=base_url)

def tool(name, schema, description=""):
    """Decorator to register a function as a tool."""
    def decorator(fn):
        fn._tool_name = name
        fn._tool_schema = schema
        fn._tool_description = description
        return fn
    return decorator
