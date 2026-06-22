"""Type stubs for the native C++ extension."""

from typing import Callable, Optional, Any, Generator

class Context:
    """Owns the async I/O context. Create once per application."""
    def __init__(self) -> None: ...

class Provider:
    """Wraps an LLM provider (Anthropic, OpenAI, etc.)."""
    def __init__(self, ctx: Context, name: str, *, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None: ...
    def model(self, model_id: str) -> "Model": ...
    def __call__(self, model_id: str) -> "Model": ...

class Model:
    """A specific language model instance."""
    ...

class ToolSet:
    """Collection of tools the agent can use."""
    def __init__(self) -> None: ...
    def add(self, name: str, schema: dict, description: str, fn: Callable[[dict], Any]) -> None: ...

class GenerateResult:
    text: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    steps: int

class StreamEvent:
    type: str  # "text_delta", "tool_call_start", "tool_call_delta", "tool_call_end", "finish", "error", "reasoning_start", "reasoning_delta", "reasoning_end", "tool_result"
    text: Optional[str]
    tool_name: Optional[str]
    tool_call_id: Optional[str]

class Agent:
    """High-level tool-loop agent."""
    def __init__(self, model: Model, tools: ToolSet, *, instructions: str = "", max_steps: int = 50) -> None: ...
    def call(self, prompt: str) -> GenerateResult: ...
    def call_stream(self, prompt: str) -> Generator[StreamEvent, None, None]: ...

def generate_text(
    model: Model,
    *,
    prompt: Optional[str] = None,
    messages: Optional[list] = None,
    system: Optional[str] = None,
    tools: Optional[ToolSet] = None,
    max_steps: int = 1,
    max_output_tokens: int = 0,
    temperature: float = -1,
) -> GenerateResult: ...

def stream_text(
    model: Model,
    *,
    prompt: Optional[str] = None,
    messages: Optional[list] = None,
    system: Optional[str] = None,
    max_output_tokens: int = 0,
    temperature: float = -1,
) -> Generator[StreamEvent, None, None]: ...

def version() -> str: ...

class Session:
    def __init__(self, agent: Agent) -> None: ...
    def send(self, prompt: str) -> GenerateResult: ...
    def send_stream(self, prompt: str, callback: Callable[[StreamEvent], None]) -> None: ...
