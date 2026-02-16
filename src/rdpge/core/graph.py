# graph.py — Graph State Manager
#
# Tracks all nodes and their states.
# Handles:
#   - Node creation
#   - Edge application (context restoration)
#   - Task/step counter updates

from typing import Optional
from .models import GraphState, NodeState, ExecutionResult

# ---- Graph State Manager ----
class GraphStateManager:

    async def update_graph(
        self,
        graph: GraphState,
        execution: ExecutionResult,
        tool_output: Optional[str],
        active_edge: Optional[str] = None
    ) -> NodeState:
        """
        Updates the graph after a step execution:
        1. Handles edge restoration
        2. Creates the new node
        """
        if execution.action:
            graph.active_node = execution.action.node

        # handling edge — edges last ONE turn, then reset
        # Now passed explicitly via active_edge (from restore_context tool)
        if active_edge:
            graph.active_edge = active_edge
        else:
            graph.active_edge = None

        # updating task and step
        task_step = self._parse_node_id(execution.action.node)
        graph.current_task = task_step[0]
        graph.current_step = task_step[1]

        node = NodeState(
            node_id=execution.action.node,
            task_id=graph.current_task,
            code=execution.code,
            console_output=execution.console_output,
            tool_output=tool_output,
            tool_call=execution.action.tool_call,
            edge=active_edge
        )

        graph.nodes[node.node_id] = node
        return node  # return node object

    def _parse_node_id(self, node_id: str) -> tuple[str, int]:
        parts = node_id.split("-")
        if len(parts) != 2 or not parts[1][1].isdigit():
            raise ValueError(f"Invalid node ID format: {node_id}")
        return (str(parts[1][0]), int(parts[1][1]))