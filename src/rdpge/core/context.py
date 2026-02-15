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
        if not graph.nodes:
            return (
                "Active Node: (awaiting first action)\n"
                "Current Task: a\n"
                "Task Map: (no tasks yet)"
            )

        # 1. Collect per-task metadata
        #    - last_step_index: position of the task's last node in the ordered dict
        #    - step_count: total nodes in this task
        task_info: dict[str, dict] = {}
        node_list = list(graph.nodes.keys())
        total_nodes = len(node_list)

        for idx, (node_id, node) in enumerate(graph.nodes.items()):
            tid = node.task_id
            if tid not in task_info:
                task_info[tid] = {"last_index": idx, "step_count": 0}
            task_info[tid]["last_index"] = idx
            task_info[tid]["step_count"] += 1

        # 2. Compute in-degree: count unique SOURCE tasks that created edges to each TARGET task
        #    edge field is like "node-a" → target task is "a"
        #    source task is the task_id of the node that declared the edge
        in_degree: dict[str, int] = {tid: 0 for tid in task_info}
        edge_sources: dict[str, set] = {tid: set() for tid in task_info}

        for node in graph.nodes.values():
            if node.edge:
                # Extract target task letter from edge (e.g., "node-a" → "a")
                target_task = node.edge.replace("node-", "")
                source_task = node.task_id
                # Only count cross-task edges (not self-edges)
                if target_task in edge_sources and source_task != target_task:
                    edge_sources[target_task].add(source_task)

        for tid in in_degree:
            in_degree[tid] = len(edge_sources[tid])

        # 3. Build task map lines
        task_lines = []
        for tid, info in task_info.items():
            if tid == graph.current_task:
                distance_str = "active"
            else:
                distance = total_nodes - 1 - info["last_index"]
                distance_str = f"{distance} steps ago"

            refs = in_degree[tid]
            refs_str = f"{refs} reference{'s' if refs != 1 else ''}"
            steps_str = f"{info['step_count']} step{'s' if info['step_count'] != 1 else ''}"

            task_lines.append(
                f"  Task {tid.upper()}: {distance_str} | {refs_str} | {steps_str}"
            )

        task_map = "\n".join(task_lines)

        manifest = f"""Active Node: {graph.active_node or "(awaiting first action)"}
Current Task: {graph.current_task}
Active Edge: {graph.active_edge or "(none)"}

Task Map:
{task_map}"""
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
        """Turn 2+: Rebuild entire messages list from scratch.

        For multi-turn sessions, this interleaves [USER TASK] markers
        with nodes based on request_index, so the LLM sees:
            [USER TASK 1] → nodes from request 1 → [USER TASK 2] → nodes from request 2 → ...
        """
        messages = []

        # 1. System prompt (fresh, with updated manifest)
        messages.append(Message(
            role="system",
            content=self._render_system_prompt(graph)
        ))

        # 2. Interleave requests with their nodes
        # Group nodes by request_index
        nodes_by_request: dict[int, list[tuple[str, any]]] = {}
        for node_id, node in graph.nodes.items():
            req_idx = node.request_index
            if req_idx not in nodes_by_request:
                nodes_by_request[req_idx] = []
            nodes_by_request[req_idx].append((node_id, node))

        # Walk through each request and its nodes
        for req_idx, request_text in enumerate(graph.requests):
            # Add user task
            messages.append(Message(
                role="user",
                content=f"[USER TASK]\n{request_text}"
            ))

            # Add nodes belonging to this request
            for node_id, node in nodes_by_request.get(req_idx, []):
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

        # 3. Include error from previous failed attempt (if any)
        if graph.last_error:
            messages.append(Message(
                role="user",
                content=f"[CODE ERROR]\n{graph.last_error}"
            ))
            graph.last_error = None  # Clear after showing once

        return messages

