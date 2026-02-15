# agent.py — Agent Entry Point
#
# The developer-facing API for RDPGE.
# This is what users import and use:
#
#   from rdpge import Agent, tool
#
#   @tool()
#   def read_file(path: str) -> str:
#       """Read file contents from disk."""
#       return open(path).read()
#
#   agent = Agent(
#       llm=OpenAIAdapter(LLMConfig(model="gpt-4o", api_key="...")),
#       tools=[read_file],
#       instructions="You are a code assistant.",
#   )
#
#   result = await agent.run("Fix the bug in auth.py")

from typing import Optional
from .core.engine import ExecutionEngine, AgentResult
from .core.context import ContextConstructor
from .tools.base import ToolWrapper, BaseTool
from .tools.registry import ToolRegistry
from .observe.hooks import HookManager


class Agent:
    """
    The RDPGE Agent — developer entry point.

    Args:
        llm: Any LLM provider with an async generate(messages) method
        tools: List of ToolWrappers (from @tool decorator, BaseTool, or ToolSpec)
        instructions: Agent instructions string for the system prompt
        instructions_file: Path to a .md file containing agent instructions
        max_steps: Maximum execution steps before forced termination (default: 25)
    """

    def __init__(
        self,
        llm,
        tools: list[ToolWrapper] = None,
        instructions: str = "",
        instructions_file: Optional[str] = None,
        max_steps: int = 25,
    ):
        self.llm = llm
        self.max_steps = max_steps

        # Build tool registry
        self.registry = ToolRegistry()
        if tools:
            for t in tools:
                # Support both ToolWrapper and BaseTool
                if isinstance(t, BaseTool):
                    self.registry.register(t.to_wrapper())
                elif isinstance(t, ToolWrapper):
                    self.registry.register(t)
                else:
                    raise TypeError(
                        f"Expected ToolWrapper or BaseTool, got {type(t).__name__}. "
                        f"Did you forget to use the @tool decorator?"
                    )

        # Build context constructor
        self.context_builder = ContextConstructor(
            domain_context=instructions,
            domain_file=instructions_file,
            tools_prompt=self.registry.to_prompt()
        )

        # Hook manager
        self.hooks = HookManager()

        # Engine
        self._engine = ExecutionEngine(
            llm=self.llm,
            registry=self.registry,
            context_builder=self.context_builder,
            hooks=self.hooks,
            max_steps=self.max_steps,
        )

    def on(self, event: str):
        """
        Decorator to register event hooks.

        Usage:
            @agent.on("step_end")
            async def log_step(data):
                print(f"Step {data['step']}: {data['node_id']}")
        """
        return self.hooks.on(event)

    async def run(self, request: str) -> AgentResult:
        """
        Run the agent with a user request.

        Args:
            request: The task to execute (e.g., "Fix the bug in auth.py")

        Returns:
            AgentResult with success status, execution trace, and graph state

        Raises:
            TypeError: If request is not a string.
            ValueError: If request is empty or whitespace-only.
        """
        if not isinstance(request, str):
            raise TypeError(
                f"Request must be a string, got {type(request).__name__}."
            )
        if not request.strip():
            raise ValueError(
                "Request cannot be empty or whitespace-only."
            )
        return await self._engine.run(request.strip())
