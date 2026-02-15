# hooks.py — Event Hook System
#
# Allows developers to plug into agent lifecycle events.
# Events: on_step_start, on_step_end, on_tool_called, on_error, on_complete
#
# Usage:
#   agent = Agent(llm=..., tools=[...])
#
#   @agent.on("step_end")
#   async def log_step(data):
#       print(f"Step {data['step']} completed: {data['node_id']}")

import asyncio
from typing import Any, Callable, Coroutine
from dataclasses import dataclass, field


# Type alias for hook callbacks
HookCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class HookManager:
    """
    Event system for agent lifecycle hooks.

    Supported events:
        - step_start:   Before LLM inference
        - step_end:     After graph update
        - tool_called:  After tool execution
        - error:        When an error occurs
        - complete:     When agent finishes
    """

    VALID_EVENTS = {"step_start", "step_end", "tool_called", "error", "complete"}

    def __init__(self):
        self._hooks: dict[str, list[HookCallback]] = {
            event: [] for event in self.VALID_EVENTS
        }

    def on(self, event: str) -> Callable:
        """
        Decorator to register an event hook.

        Usage:
            @hooks.on("step_end")
            async def my_handler(data):
                print(data)
        """
        if event not in self.VALID_EVENTS:
            raise ValueError(
                f"Unknown event '{event}'. "
                f"Valid events: {', '.join(self.VALID_EVENTS)}"
            )

        def decorator(fn: HookCallback) -> HookCallback:
            self._hooks[event].append(fn)
            return fn

        return decorator

    def register(self, event: str, callback: HookCallback) -> None:
        """Register a hook callback programmatically."""
        if event not in self.VALID_EVENTS:
            raise ValueError(f"Unknown event '{event}'.")
        self._hooks[event].append(callback)

    async def emit(self, event: str, data: dict[str, Any]) -> None:
        """
        Emit an event, calling all registered hooks.
        Hooks are called concurrently. Errors in hooks are caught
        and do NOT crash the agent.
        """
        if event not in self.VALID_EVENTS:
            return

        callbacks = self._hooks[event]
        if not callbacks:
            return

        # Run all hooks concurrently, catch errors
        results = await asyncio.gather(
            *(cb(data) for cb in callbacks),
            return_exceptions=True
        )

        # Log hook errors but don't propagate them
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(
                    f"[RDPGE WARNING] Hook error on '{event}': "
                    f"{type(result).__name__}: {result}"
                )
