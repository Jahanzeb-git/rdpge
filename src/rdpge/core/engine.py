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
from typing import Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from .models import GraphState, Message, ExecutionResult
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
    """

    def __init__(
        self,
        llm,  # LLMProvider (any object with async generate(messages))
        registry: ToolRegistry,
        context_builder: ContextConstructor,
        hooks: HookManager,
        max_steps: int = 25,
    ):
        self.llm = llm
        self.registry = registry
        self.context_builder = context_builder
        self.hooks = hooks
        self.max_steps = max_steps
        self.executor = CodeExecutor()
        self.graph_manager = GraphStateManager()

    async def run(self, request: str) -> AgentResult:
        """
        Run the full agent loop for a given user request.

        Args:
            request: The user's task (e.g., "Fix the auth bug in auth.py")

        Returns:
            AgentResult with success status, trace, and final graph state
        """
        # Initialize session
        session_id = str(uuid.uuid4())[:8]
        graph = GraphState(
            session_id=session_id,
            original_request=request
        )
        trace = SessionTrace(
            session_id=session_id,
            original_request=request
        )

        step = 0

        try:
            while step < self.max_steps:
                step += 1
                step_start = time.time()

                # --- Emit step_start hook ---
                await self.hooks.emit("step_start", {
                    "step": step,
                    "session_id": session_id,
                    "active_node": graph.active_node,
                })

                # --- 1. Build context ---
                if step == 1:
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

                # --- 5. Route tool call (if any) ---
                tool_output = None
                tool_name = None

                if execution.action and execution.action.tool_call:
                    tool_name = execution.action.tool_call.name
                    tool_output = await self.registry.execute_tool(
                        execution.action.tool_call
                    )
                    await self.hooks.emit("tool_called", {
                        "step": step,
                        "tool": tool_name,
                        "args": execution.action.tool_call.args,
                        "output_length": len(tool_output),
                    })

                # --- 6. Update graph ---
                node = await self.graph_manager.update_graph(
                    graph, execution, tool_output
                )

                # --- 7. Record trace ---
                step_duration = (time.time() - step_start) * 1000
                trace.add_step(StepTrace(
                    step_number=step,
                    node_id=node.node_id,
                    task_id=node.task_id,
                    reason=execution.action.reason if execution.action else "",
                    tool_name=tool_name,
                    tool_args=execution.action.tool_call.args if execution.action and execution.action.tool_call else None,
                    tool_output_length=len(tool_output) if tool_output else 0,
                    console_output_length=len(execution.console_output),
                    code_length=len(code),
                    edge_restored=execution.action.edge if execution.action else None,
                    success=True,
                    duration_ms=step_duration,
                ))

                # --- Emit step_end hook ---
                await self.hooks.emit("step_end", {
                    "step": step,
                    "node_id": node.node_id,
                    "task_id": node.task_id,
                    "reason": execution.action.reason if execution.action else "",
                    "tool": tool_name,
                    "duration_ms": step_duration,
                })

                # --- 8. Check completion ---
                if execution.action and execution.action.tool_call is None:
                    # LLM signaled completion (tool_call is None)
                    trace.finalize()
                    await self.hooks.emit("complete", {
                        "session_id": session_id,
                        "steps": step,
                        "reason": execution.action.reason,
                        "summary": trace.summary(),
                    })
                    return AgentResult(
                        success=True,
                        reason=execution.action.reason,
                        steps=step,
                        session_id=session_id,
                        trace=trace,
                        graph=graph,
                    )

            # Max steps reached without completion
            trace.finalize()
            return AgentResult(
                success=False,
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
