# context.py — Context Constructor
#
# Builds the list[Message] for each LLM inference call.
# Handles:
#   - System prompt assembly (Jinja2 template rendering)
#   - Graph manifest generation
#   - Node context rendering (with blurring applied)
#   - Message list reconstruction each turn

from typing import Optional
from jinja2 import Template
from pathlib import Path
from .models import GraphState, Message


class ContextConstructor:
    def __init__(
        self,
        domain_context: str = "",
        domain_file: Optional[str] = None,
        tools_prompt: str = "",
    ):
        self.tools_prompt = tools_prompt

        # Resolve domain context: file takes priority over string
        if domain_file:
            path = Path(domain_file)
            if not path.exists():
                raise FileNotFoundError(f"Domain file not found: {domain_file}")
            self.domain_context = path.read_text(encoding="utf-8")
        else:
            self.domain_context = domain_context

        # Load the internal Jinja2 system prompt template
        template_path = Path(__file__).parent.parent / "prompts" / "templates" / "system.j2"
        self.template = Template(template_path.read_text(encoding="utf-8"))

    # --- Build manifest section ---
    def _build_manifest(self, graph: GraphState) -> str:
        # Collect tasks that are NOT the current task
        other_tasks = set()
        for node in graph.nodes.values():
            if node.task_id != graph.current_task:
                other_tasks.add(node.task_id)

        manifest = f"""## GRAPH MANIFEST
Active Node: {graph.active_node or "(awaiting first action)"}
Current Task: {graph.current_task}
Inactive Tasks: {list(other_tasks) if other_tasks else "(none yet)"}
Active Edge: {graph.active_edge or "(none)"}

You may create an edge to any node from an inactive task to restore its tool output."""
        return manifest

    # --- Render system prompt ---
    def _render_system_prompt(self, graph: GraphState) -> str:
        return self.template.render(
            available_tools=self.tools_prompt,
            domain_context=self.domain_context,
            graph_manifest=self._build_manifest(graph)
        )

    # --- Decide if tool output is visible ---
    def _is_visible(self, node_task: str, node_id: str, graph: GraphState) -> bool:
        # Same task → always visible
        if node_task == graph.current_task:
            return True
        # Edge restores ENTIRE task — "node-a" makes ALL task "a" nodes visible
        if graph.active_edge and graph.active_edge == f"node-{node_task}":
            return True
        # Otherwise → blurred
        return False

    # --- Build node's execution result content ---
    def _build_node_result(self, node_id: str, node, graph: GraphState) -> str:
        console = node.console_output or "(none)"

        if self._is_visible(node.task_id, node_id, graph):
            tool = node.tool_output or "(no tool called)"
        else:
            tool = f"[BLURRED — this task's context is hidden. You can create an edge to restore it.]"

        return f"""[EXECUTION RESULT: {node_id}]
Console: {console}
Tool: {tool}
[END RESULT]"""

    # =============================================
    # PUBLIC METHODS — Called by the Engine
    # =============================================

    def build_initial_context(self, graph: GraphState) -> list[Message]:
        """Turn 1: No nodes exist yet. Just system + user task."""
        messages = []

        # 1. System prompt
        messages.append(Message(
            role="system",
            content=self._render_system_prompt(graph)
        ))

        # 2. User task
        messages.append(Message(
            role="user",
            content=f"[USER TASK]\n{graph.original_request}"
        ))

        # 3. Include error from previous failed attempt (if any)
        if graph.last_error:
            messages.append(Message(
                role="user",
                content=f"[CODE ERROR]\n{graph.last_error}"
            ))
            graph.last_error = None  # Clear after showing once

        return messages

    def build_turn_context(self, graph: GraphState) -> list[Message]:
        """Turn 2+: Rebuild entire messages list from scratch."""
        messages = []

        # 1. System prompt (fresh, with updated manifest)
        messages.append(Message(
            role="system",
            content=self._render_system_prompt(graph)
        ))

        # 2. User task (from graph, never lost)
        messages.append(Message(
            role="user",
            content=f"[USER TASK]\n{graph.original_request}"
        ))

        # 3. Loop through ALL nodes in order
        for node_id, node in graph.nodes.items():
            # Assistant message: the code LLM generated
            messages.append(Message(
                role="assistant",
                content=node.code
            ))

            # User message: execution result (with blurring applied)
            messages.append(Message(
                role="user",
                content=self._build_node_result(node_id, node, graph)
            ))

        # 4. Include error from previous failed attempt (if any)
        if graph.last_error:
            messages.append(Message(
                role="user",
                content=f"[CODE ERROR]\n{graph.last_error}"
            ))
            graph.last_error = None  # Clear after showing once

        return messages
