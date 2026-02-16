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
from .storage.base import InMemoryStore, deserialize_graph


class Agent:
    """
    The RDPGE Agent — developer entry point.

    Supports multi-turn execution: call run() multiple times
    and the agent preserves context within the same session.

    Args:
        llm: Any LLM provider with an async generate(messages) method
        tools: List of ToolWrappers (from @tool decorator, BaseTool, or ToolSpec)
        instructions: Agent instructions string for the system prompt
        instructions_file: Path to a .md file containing agent instructions
        max_steps: Maximum execution steps per run (default: 25)
        store: SessionStore backend for persistence (default: InMemoryStore)
        signal_handlers: Override built-in signal tool behavior. Dict mapping
            signal name to async handler function.
            Example: {"ask_user": my_ask_handler, "complete": my_complete_handler}
    """

    def __init__(
        self,
        llm,
        tools: list[ToolWrapper] = None,
        instructions: str = "",
        instructions_file: Optional[str] = None,
        max_steps: int = 25,
        store=None,
        signal_handlers: dict = None,
    ):
        self.llm = llm
        self.max_steps = max_steps
        self.store = store or InMemoryStore()
        self.signal_handlers = signal_handlers or {}

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

        # Engine (with store for persistence and signal handlers)
        self._engine = ExecutionEngine(
            llm=self.llm,
            registry=self.registry,
            context_builder=self.context_builder,
            hooks=self.hooks,
            store=self.store,
            max_steps=self.max_steps,
            signal_handlers=self.signal_handlers,
        )

    # ---- Session Management ----

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID, or None if no session is active."""
        return self._engine.session_id

    def on(self, event: str):
        """
        Decorator to register event hooks.

        Usage:
            @agent.on("step_end")
            async def log_step(data):
                print(f"Step {data['step']}: {data['node_id']}")
        """
        return self.hooks.on(event)

    def new_session(self) -> None:
        """
        Start a fresh session. Clears all in-memory state.

        Previous session data remains in the store (if one is configured)
        and can be resumed later via load_session().
        """
        self._engine.reset()

    def abort(self, reason: str = "User aborted execution") -> None:
        """
        Abort the currently running execution.

        Call this from a hook callback, a UI handler, or another
        coroutine to forcefully stop the agent at the next loop iteration.
        The abort reason is recorded in the graph so the LLM knows
        about it on the next turn.

        Args:
            reason: Human-readable reason for the abort
        """
        self._engine.request_abort(reason)

    async def load_session(self, session_id: str) -> bool:
        """
        Load a previously saved session from the store.

        Args:
            session_id: The session ID to load

        Returns:
            True if session was loaded, False if not found
        """
        state = await self.store.load(session_id)
        if state is None:
            return False

        graph = deserialize_graph(state)
        self._engine.set_graph(graph)
        return True

    async def list_sessions(self) -> list[str]:
        """List all session IDs in the store."""
        return await self.store.list_sessions()

    async def run(self, request: str) -> AgentResult:
        """
        Run the agent with a user request.

        First call creates a new session. Subsequent calls
        continue the same session with full context.

        Use new_session() to start fresh, or
        load_session() to resume a previous session.

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

