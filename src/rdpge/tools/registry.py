# registry.py — Tool Registry
#
# Collects tools, validates names, generates prompt sections,
# and routes tool calls to the correct tool.

from typing import Any, Optional
from .base import ToolWrapper, BaseTool, ToolSpec
from ..core.models import ToolCall


class ToolRegistry:
    """
    Central registry for all tools available to the agent.

    Usage:
        registry = ToolRegistry()
        registry.register(read_file_tool)     # ToolWrapper from @tool
        registry.register(db_tool.to_wrapper()) # From BaseTool
        registry.register(spec.to_wrapper())    # From ToolSpec

        # Generate prompt section for system prompt
        prompt = registry.to_prompt()

        # Execute a tool call from the LLM
        output = await registry.execute_tool(tool_call)
    """

    def __init__(self):
        self._tools: dict[str, ToolWrapper] = {}

    def register(self, tool: ToolWrapper) -> None:
        """Register a tool. Raises if name already taken."""
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                f"Each tool must have a unique name."
            )
        self._tools[tool.name] = tool

    def register_many(self, tools: list[ToolWrapper]) -> None:
        """Register multiple tools at once."""
        for t in tools:
            self.register(t)

    def get(self, name: str) -> Optional[ToolWrapper]:
        """Get a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def to_prompt(self) -> str:
        """
        Generate the AVAILABLE TOOLS section for the system prompt.

        Returns something like:
            ## AVAILABLE TOOLS
            - read_file(path: str) — Read file contents from disk
            - write_file(path: str, content: str) — Write content to a file
        """
        if not self._tools:
            return "## AVAILABLE TOOLS\n(no tools registered)"

        lines = ["## AVAILABLE TOOLS"]
        for tool in self._tools.values():
            lines.append(tool.to_prompt())
        return "\n".join(lines)

    async def execute_tool(self, tool_call: ToolCall) -> str:
        """
        Route a ToolCall to the correct tool and return the output.

        Args:
            tool_call: ToolCall from the LLM's action dict

        Returns:
            Tool output as a string, or error message if tool not found.
        """
        tool = self._tools.get(tool_call.name)
        if tool is None:
            return (
                f"[TOOL ERROR] Unknown tool '{tool_call.name}'. "
                f"Available: {', '.join(self._tools.keys())}"
            )
        return await tool.execute(tool_call.args)
