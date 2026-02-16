# context.py — Context Constructor
#
# Builds the list[Message] for each LLM inference call.
# Handles:
#   - System prompt assembly (Jinja2 template rendering)
#   - Graph manifest generation
#   - Node context rendering (with blurring applied)
#   - Message list reconstruction each turn

from typing import Optional
import json
from jinja2 import Template
from pathlib import Path
from .models import GraphState, Message


class ContextConstructor:
    def __init__(
        self,
        domain_context: str = "",
        domain_file: Optional[str] = None,
        tools_prompt: str = "",
        max_steps: int = 25,
    ):
        self.tools_prompt = tools_prompt
        self.max_steps = max_steps

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
            manifest_data = {
                "tasks": {},
                "edges": [],
                "runtime": {
                    "step": 0,
                    "active_node": None,
                    "current_task": graph.current_task,
                    "restored_context": None
                }
            }
            return json.dumps(manifest_data, indent=2)

        # 1. Analyze Task States & Edges
        task_stats: dict[str, int] = {}  # task_id -> step_count
        edges: set[str] = set()          # "source -> target" strings

        for node in graph.nodes.values():
            # Count steps per task
            tid = node.task_id
            task_stats[tid] = task_stats.get(tid, 0) + 1

            # Detect edges (dependencies)
            # If node in task B restores task A context, that's B -> A dependency
            if node.edge:
                # node.edge format is "node-a" -> target is "a"
                target_task = node.edge.replace("node-", "")
                if target_task != tid:  # exclude self-references
                    edges.add(f"{tid} -> {target_task}")

        # 2. Compute in-degree (References)
        in_degree: dict[str, int] = {tid: 0 for tid in task_stats}
        for edge_str in edges:
            # edge string is "source -> target"
            _, target = edge_str.split(" -> ")
            in_degree[target] = in_degree.get(target, 0) + 1

        # 3. Build task map with full metrics
        total_nodes = len(graph.nodes)
        tasks_map = {}
        
        # We need the last index of each task to compute distance
        task_last_index = {}
        for idx, node in enumerate(graph.nodes.values()):
            task_last_index[node.task_id] = idx

        for tid, count in task_stats.items():
            status = "active" if tid == graph.current_task else "inactive"
            
            # Distance: steps since last activity
            # If active, distance is 0 (or "active")
            if tid == graph.current_task:
                 distance = 0
            else:
                 last_idx = task_last_index.get(tid, 0)
                 distance = total_nodes - 1 - last_idx

            tasks_map[tid] = {
                "status": status,
                "steps": count,
                "distance": distance,
                "references": in_degree.get(tid, 0)
            }

        # 3. Build Runtime State
        current_step = len(graph.nodes) + 1
        
        manifest_data = {
            "tasks": tasks_map,
            "edges": sorted(list(edges)),
            "runtime": {
                "step": current_step,
                "current_task": graph.current_task,
                "active_node": graph.active_node,
                "restored_context": graph.active_edge or None
            }
        }

        return json.dumps(manifest_data, indent=2)

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
                # IMPORTANT: Wrap in ```python``` fences so the LLM sees
                # its own history in the format we expect it to produce.
                # Without this, the LLM pattern-matches its fenceless
                # history and stops using fences.
                messages.append(Message(
                    role="assistant",
                    content=f"```python\n{node.code}\n```"
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

