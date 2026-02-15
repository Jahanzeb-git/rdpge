# base.py — Tool Definition System
#
# Provides:
#   - @tool decorator: wraps plain functions into ToolWrapper
#   - ToolWrapper: uniform internal representation
#   - BaseTool: abstract class for advanced tools with lifecycle
#   - ToolSpec: config-driven tool creation

import inspect
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class ToolParam:
    """Describes a single parameter of a tool."""
    name: str
    type_hint: str
    description: str = ""
    required: bool = True
    default: Any = None


class ToolWrapper:
    """
    Uniform internal representation of a tool.
    Created by the @tool decorator or from BaseTool/ToolSpec.
    """
    def __init__(
        self,
        name: str,
        description: str,
        params: list[ToolParam],
        fn: Callable,
        is_async: bool = False
    ):
        self.name = name
        self.description = description
        self.params = params
        self.fn = fn
        self.is_async = is_async

    async def execute(self, args: dict[str, Any]) -> str:
        """Execute the tool and return the output as a string."""
        try:
            if self.is_async:
                result = await self.fn(**args)
            else:
                result = self.fn(**args)
            return str(result)
        except Exception as e:
            return f"[TOOL ERROR] {self.name}: {str(e)}"

    def to_prompt(self) -> str:
        """Generate the prompt description for this tool."""
        params_str = ", ".join(
            f"{p.name}: {p.type_hint}" for p in self.params
        )
        return f"- {self.name}({params_str}) — {self.description}"


def _extract_params(fn: Callable) -> list[ToolParam]:
    """Extract parameters from a function's signature and type hints."""
    sig = inspect.signature(fn)
    hints = fn.__annotations__
    params = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        type_hint = hints.get(param_name, "Any")
        if hasattr(type_hint, "__name__"):
            type_hint = type_hint.__name__
        else:
            type_hint = str(type_hint)

        has_default = param.default is not inspect.Parameter.empty

        params.append(ToolParam(
            name=param_name,
            type_hint=type_hint,
            required=not has_default,
            default=param.default if has_default else None
        ))

    return params


def tool(description: str = ""):
    """
    Decorator that wraps a plain function into a ToolWrapper.

    Description priority:
        1. Explicit description parameter (if provided)
        2. Function's docstring
        3. Raises ValueError (no description = LLM can't understand the tool)

    Usage:
        @tool()
        def read_file(path: str) -> str:
            \"\"\"Read file contents from disk and return as text.\"\"\"
            with open(path) as f:
                return f.read()

        # Or with explicit description (overrides docstring):
        @tool("Read file contents from disk")
        def read_file(path: str) -> str:
            ...
    """
    MAX_DESCRIPTION_LENGTH = 200

    def decorator(fn: Callable) -> ToolWrapper:
        # Resolve description: explicit > docstring > error
        resolved = description or (fn.__doc__ or "").strip()
        if not resolved:
            raise ValueError(
                f"Tool '{fn.__name__}' has no description. "
                f"Add a docstring or pass description to @tool()."
            )
        if len(resolved) > MAX_DESCRIPTION_LENGTH:
            resolved = resolved[:MAX_DESCRIPTION_LENGTH] + "..."

        params = _extract_params(fn)
        is_async = inspect.iscoroutinefunction(fn)

        return ToolWrapper(
            name=fn.__name__,
            description=resolved,
            params=params,
            fn=fn,
            is_async=is_async
        )

    return decorator


class BaseTool(ABC):
    """
    Abstract base for advanced tools that need setup/teardown.

    Usage:
        class DatabaseTool(BaseTool):
            name = "query_db"
            description = "Run a SQL query"

            async def setup(self):
                self.conn = await connect_db()

            async def execute(self, query: str) -> str:
                return await self.conn.execute(query)

            async def teardown(self):
                await self.conn.close()
    """
    name: str = ""
    description: str = ""

    async def setup(self) -> None:
        """Called once before first use. Override for initialization."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool. Must be implemented by subclasses."""
        ...

    async def teardown(self) -> None:
        """Called on agent shutdown. Override for cleanup."""
        pass

    def to_wrapper(self) -> ToolWrapper:
        """Convert to ToolWrapper for registry compatibility."""
        params = _extract_params(self.execute)
        return ToolWrapper(
            name=self.name,
            description=self.description,
            params=params,
            fn=self.execute,
            is_async=True
        )


@dataclass
class ToolSpec:
    """
    Config-driven tool creation (for dynamic tool loading).

    Usage:
        spec = ToolSpec(
            name="read_file",
            description="Read file contents",
            params=[ToolParam("path", "str")],
            fn=lambda path: open(path).read()
        )
        wrapper = spec.to_wrapper()
    """
    name: str
    description: str
    params: list[ToolParam] = field(default_factory=list)
    fn: Optional[Callable] = None
    is_async: bool = False

    def to_wrapper(self) -> ToolWrapper:
        if self.fn is None:
            raise ValueError(f"ToolSpec '{self.name}' has no function assigned.")
        return ToolWrapper(
            name=self.name,
            description=self.description,
            params=self.params,
            fn=self.fn,
            is_async=self.is_async
        )
