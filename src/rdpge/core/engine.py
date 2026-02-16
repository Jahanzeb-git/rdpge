# engine.py — Execution Engine
#
# The main orchestration loop. This is the HEART of RDPGE.
# It coordinates all components in sequence:
#   1. Build context (ContextConstructor)
#   2. Call LLM (LLMProvider)
#   3. Extract code from response
#   4. Execute code (CodeExecutor)
#   5. Route tool calls (ToolRegistry)
#   6. Update graph (GraphStateManager)
#   7. Emit hooks + record trace
#   8. Check completion → loop or return

import re
import json
import time
import uuid
from typing import Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from .models import GraphState, Message, ExecutionResult, SIGNAL_TOOLS, ToolCall
from .executor import CodeExecutor
from .graph import GraphStateManager
from .context import ContextConstructor
from ..tools.registry import ToolRegistry
from ..observe.trace import StepTrace, SessionTrace
from ..observe.hooks import HookManager


@dataclass
class AgentResult:
    """Final result returned to the developer after agent completes."""
    success: bool
    status: str  # "completed", "awaiting_input", "surrendered", "aborted", "max_steps", "error"
    reason: str
    steps: int
    session_id: str
    trace: SessionTrace
    graph: GraphState
    error: Optional[str] = None

    def export_graph(self, path: str) -> str:
        """
        Export the full execution graph as a JSON file.

        The JSON contains every node with its code, tool output,
        console output, edges, and timestamps — complete observability.

        Args:
            path: File path to write the JSON (e.g., "./output/session.json")

        Returns:
            The absolute path of the created file.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the graph data structure
        graph_data = {
            "session_id": self.graph.session_id,
            "original_request": self.graph.original_request,
            "success": self.success,
            "reason": self.reason,
            "total_steps": self.steps,
            "current_task": self.graph.current_task,
            "current_step": self.graph.current_step,
            "active_node": self.graph.active_node,
            "active_edge": self.graph.active_edge,
            "nodes": {},
            "trace": asdict(self.trace),
        }

        # Serialize each node
        for node_id, node in self.graph.nodes.items():
            graph_data["nodes"][node_id] = {
                "node_id": node.node_id,
                "task_id": node.task_id,
                "code": node.code,
                "console_output": node.console_output,
                "tool_output": node.tool_output,
                "tool_call": node.tool_call.model_dump() if node.tool_call else None,
                "edge": node.edge,
                "timestamp": node.timestamp,
            }

        output_path.write_text(
            json.dumps(graph_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        return str(output_path.resolve())


class ExecutionEngine:
    """
    The RDPGE execution engine.

    Orchestrates the full agent loop:
        context → LLM → extract code → execute → tool → graph update → repeat

    Supports multi-turn execution: the engine keeps its GraphState alive
    between run() calls. Each call either creates a new session (first run)
    or continues the existing session (continuation).
    """

    def __init__(
        self,
        llm,  # LLMProvider (any object with async generate(messages))
        registry: ToolRegistry,
        context_builder: ContextConstructor,
        hooks: HookManager,
        store=None,  # Optional SessionStore
        max_steps: int = 25,
        signal_handlers: dict = None,  # Developer overrides for signal tools
    ):
        self.llm = llm
        self.registry = registry
        self.context_builder = context_builder
        self.hooks = hooks
        self.store = store
        self.max_steps = max_steps
        self.executor = CodeExecutor()
        self.graph_manager = GraphStateManager()

        # Default signal handlers — developers can override via Agent
        self.signal_handlers = signal_handlers or {}

        # Abort mechanism — developer-side, not LLM-side
        self._abort_requested = False
        self._abort_reason = ""

        # Multi-turn state — persists between run() calls
        self.graph: Optional[GraphState] = None
        self.session_id: Optional[str] = None

    def reset(self) -> None:
        """Clear session state for a fresh start."""
        self.graph = None
        self.session_id = None
        self._abort_requested = False
        self._abort_reason = ""

    def set_graph(self, graph: GraphState) -> None:
        """Set graph state (used by load_session)."""
        self.graph = graph
        self.session_id = graph.session_id

    async def _save_state(self) -> None:
        """Save current graph state to store (if store is provided)."""
        if self.store and self.graph:
            from ..storage.base import serialize_graph
            await self.store.save(self.session_id, serialize_graph(self.graph))

    def request_abort(self, reason: str = "User aborted execution") -> None:
        """
        Request the engine to abort at the next loop iteration.

        This is a developer-side mechanism — called from outside
        the engine (e.g., from a UI button or hook callback).
        The abort is recorded in graph.last_error so the LLM
        sees it on the next turn.

        Args:
            reason: Human-readable reason for the abort
        """
        self._abort_requested = True
        self._abort_reason = reason

    async def run(self, request: str) -> AgentResult:
        """
        Run the full agent loop for a given user request.

        If no session exists, creates a new one.
        If a session exists, continues it with the new request.

        Args:
            request: The user's task (e.g., "Fix the auth bug in auth.py")

        Returns:
            AgentResult with success status, trace, and final graph state
        """
        # Determine: new session or continuation?
        if self.graph is None:
            # First run — create new session
            session_id = str(uuid.uuid4())[:8]
            self.session_id = session_id
            self.graph = GraphState(
                session_id=session_id,
                original_request=request,
                requests=[request],
            )
        else:
            # Continuation — append new request to existing session
            session_id = self.session_id
            self.graph.requests.append(request)
            self.graph.last_error = None  # Clear errors from previous run

        graph = self.graph
        request_index = len(graph.requests) - 1

        trace = SessionTrace(
            session_id=session_id,
            original_request=request
        )

        step = 0
        self._abort_requested = False  # Reset abort flag at start of each run

        try:
            while step < self.max_steps:
                step += 1
                step_start = time.time()

                # --- Check abort flag (set by developer via agent.abort()) ---
                if self._abort_requested:
                    # Record in graph so LLM knows in next turn
                    graph.last_error = f"[USER ABORT] {self._abort_reason}"
                    await self._save_state()
                    trace.finalize()
                    await self.hooks.emit("abort", {
                        "session_id": session_id,
                        "step": step,
                        "reason": self._abort_reason,
                    })
                    if "abort" in self.signal_handlers:
                        await self.signal_handlers["abort"](self._abort_reason)
                    return AgentResult(
                        success=False,
                        status="aborted",
                        reason=self._abort_reason,
                        steps=step - 1,  # Don't count the aborted step
                        session_id=session_id,
                        trace=trace,
                        graph=graph,
                    )

                # --- Emit step_start hook ---
                await self.hooks.emit("step_start", {
                    "step": step,
                    "session_id": session_id,
                    "active_node": graph.active_node,
                })

                # --- 1. Build context ---
                if not graph.nodes:
                    messages = self.context_builder.build_initial_context(graph)
                else:
                    messages = self.context_builder.build_turn_context(graph)

                # --- 2. Call LLM ---
                raw_response = await self.llm.generate(messages)

                # --- 3. Extract code from response ---
                code = self._extract_code(raw_response)
                if code is None:
                    # LLM didn't return a code block — feed error back
                    error_msg = "No Python code block found in your response. Your ENTIRE output must be a ```python``` code block."
                    graph.last_error = error_msg
                    await self.hooks.emit("error", {
                        "step": step,
                        "error": error_msg,
                        "raw_response": raw_response[:500],
                    })
                    trace.add_step(StepTrace(
                        step_number=step,
                        node_id="unknown",
                        task_id=graph.current_task,
                        reason="(no code extracted)",
                        success=False,
                        error=error_msg,
                        duration_ms=(time.time() - step_start) * 1000,
                    ))
                    continue

                # --- 4. Execute code in sandbox ---
                execution = self.executor.execute(code)

                if not execution.success:
                    # Code failed — feed error + console back to LLM
                    error_detail = f"Error: {execution.error}"
                    if execution.console_output:
                        error_detail += f"\nConsole output before crash: {execution.console_output}"
                    graph.last_error = error_detail
                    await self.hooks.emit("error", {
                        "step": step,
                        "error": execution.error,
                        "code": code[:300],
                    })
                    trace.add_step(StepTrace(
                        step_number=step,
                        node_id="unknown",
                        task_id=graph.current_task,
                        reason="(execution failed)",
                        success=False,
                        error=execution.error,
                        duration_ms=(time.time() - step_start) * 1000,
                    ))
                    continue

                # --- 5. Route action ---
                tool_output = None
                tool_name = None
                active_edge = None

                if execution.action:
                    tool_name = execution.action.tool_call.name
                    args = execution.action.tool_call.args

                    # --- 5a. Handle specific tools ---

                    if tool_name == "restore_context":
                        # Restore context signal -> Set active_edge
                        # Format: "node-{task}"
                        target_task = args.get("task", "")
                        # Simple validation: ensure it looks like a task ID (usually single letter)
                        active_edge = f"node-{target_task}"
                        tool_output = f"Status: Success. Context for task '{target_task}' is now restored and visible."

                    elif tool_name in SIGNAL_TOOLS:
                        # Other signals (complete, ask_user, surrender)
                        # No immediate tool output needed, they return control
                        pass

                    else:
                        # Regular tool -> Execute
                        tool_output = await self.registry.execute_tool(
                            execution.action.tool_call
                        )
                        await self.hooks.emit("tool_called", {
                            "step": step,
                            "tool": tool_name,
                            "args": args,
                            "output_length": len(tool_output),
                        })

                    # --- 6. Update graph ---
                    # Pass active_edge (if any) to graph manager
                    node = await self.graph_manager.update_graph(
                        graph, execution, tool_output, active_edge=active_edge
                    )
                    node.request_index = request_index
                    await self._save_state()

                    # --- 7. Record trace ---
                    step_duration = (time.time() - step_start) * 1000
                    trace.add_step(StepTrace(
                        step_number=step,
                        node_id=node.node_id,
                        task_id=node.task_id,
                        reason=execution.action.reason,
                        tool_name=tool_name,
                        tool_args=args,
                        tool_output_length=len(tool_output) if tool_output else 0,
                        console_output_length=len(execution.console_output),
                        code_length=len(code),
                        edge_restored=active_edge, # Record the edge if restored
                        success=True,
                        duration_ms=step_duration,
                    ))

                    await self.hooks.emit("step_end", {
                        "step": step,
                        "node_id": node.node_id,
                        "task_id": node.task_id,
                        "reason": execution.action.reason,
                        "tool": tool_name,
                        "duration_ms": step_duration,
                    })

                    # --- 8. Handle Terminal Signals ---
                    if tool_name == "complete":
                        trace.finalize()
                        if "complete" in self.signal_handlers:
                            await self.signal_handlers["complete"](execution.action.reason)
                        await self.hooks.emit("complete", {
                            "session_id": session_id,
                            "steps": step,
                            "reason": execution.action.reason,
                            "summary": trace.summary(),
                        })
                        return AgentResult(
                            success=True,
                            status="completed",
                            reason=execution.action.reason,
                            steps=step,
                            session_id=session_id,
                            trace=trace,
                            graph=graph,
                        )

                    elif tool_name == "ask_user":
                        trace.finalize()
                        question = args.get("question", execution.action.reason)
                        if "ask_user" in self.signal_handlers:
                            await self.signal_handlers["ask_user"](question)
                        await self.hooks.emit("ask_user", {
                            "session_id": session_id,
                            "step": step,
                            "question": question,
                        })
                        return AgentResult(
                            success=True,
                            status="awaiting_input",
                            reason=question,
                            steps=step,
                            session_id=session_id,
                            trace=trace,
                            graph=graph,
                        )

                    elif tool_name == "surrender":
                        trace.finalize()
                        surrender_reason = args.get("reason", execution.action.reason)
                        if "surrender" in self.signal_handlers:
                            await self.signal_handlers["surrender"](surrender_reason)
                        await self.hooks.emit("surrender", {
                            "session_id": session_id,
                            "step": step,
                            "reason": surrender_reason,
                        })
                        return AgentResult(
                            success=False,
                            status="surrendered",
                            reason=surrender_reason,
                            steps=step,
                            session_id=session_id,
                            trace=trace,
                            graph=graph,
                        )
                    
                    # If restore_context or regular tool, continue loop naturally

            # Max steps reached without completion
            trace.finalize()
            return AgentResult(
                success=False,
                status="max_steps",
                reason=f"Agent reached maximum step limit ({self.max_steps})",
                steps=step,
                session_id=session_id,
                trace=trace,
                graph=graph,
                error=f"Max steps ({self.max_steps}) exceeded",
            )

        except Exception as e:
            trace.finalize()
            await self.hooks.emit("error", {
                "step": step,
                "error": str(e),
                "fatal": True,
            })
            return AgentResult(
                success=False,
                status="error",
                reason=f"Fatal error: {str(e)}",
                steps=step,
                session_id=session_id,
                trace=trace,
                graph=graph,
                error=str(e),
            )

    def _extract_code(self, response: str) -> Optional[str]:
        """
        Extract Python code from the LLM's response.

        Looks for ```python ... ``` blocks. Returns the code
        inside the first block, or None if no block found.
        """
        pattern = r"```python\s*\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
