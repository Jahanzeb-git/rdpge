# base.py — Session Store Protocol & InMemoryStore
#
# Provides:
#   - SessionStore: Protocol that any storage backend must implement
#   - InMemoryStore: Built-in dict-based store (dev/prototyping)
#   - serialize_graph / deserialize_graph: Conversion helpers

from typing import Protocol, runtime_checkable, Optional
from ..core.models import GraphState, NodeState, ToolCall


@runtime_checkable
class SessionStore(Protocol):
    """
    Protocol for session persistence backends.

    Any object implementing these 4 async methods can be used
    as a storage backend for RDPGE agent sessions.

    Built-in:
        InMemoryStore — dict-based, dies with process

    Developer-provided (examples):
        PostgresStore, RedisStore, S3Store, SQLiteStore

    Usage:
        store = InMemoryStore()
        agent = Agent(llm=..., tools=[...], store=store)
    """

    async def save(self, session_id: str, state: dict) -> None:
        """Save session state. Overwrites if exists."""
        ...

    async def load(self, session_id: str) -> Optional[dict]:
        """Load session state. Returns None if not found."""
        ...

    async def delete(self, session_id: str) -> None:
        """Delete a session."""
        ...

    async def list_sessions(self) -> list[str]:
        """List all stored session IDs."""
        ...


class InMemoryStore:
    """
    Dict-based session store. Zero config, dies with process.

    Suitable for:
        - Local development
        - Prototyping
        - Scripts
        - Testing

    NOT suitable for production (state lost on process restart).
    """

    def __init__(self):
        self._sessions: dict[str, dict] = {}

    async def save(self, session_id: str, state: dict) -> None:
        self._sessions[session_id] = state

    async def load(self, session_id: str) -> Optional[dict]:
        return self._sessions.get(session_id)

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())


# ---- Serialization Helpers ----

def serialize_graph(graph: GraphState) -> dict:
    """Convert GraphState to a plain dict for storage."""
    nodes_data = {}
    for node_id, node in graph.nodes.items():
        nodes_data[node_id] = {
            "node_id": node.node_id,
            "task_id": node.task_id,
            "code": node.code,
            "console_output": node.console_output,
            "tool_output": node.tool_output,
            "tool_call": node.tool_call.model_dump() if node.tool_call else None,
            "edge": node.edge,
            "timestamp": node.timestamp,
            "request_index": node.request_index,
        }

    return {
        "session_id": graph.session_id,
        "original_request": graph.original_request,
        "requests": graph.requests,
        "current_task": graph.current_task,
        "current_step": graph.current_step,
        "active_node": graph.active_node,
        "active_edge": graph.active_edge,
        "nodes": nodes_data,
    }


def deserialize_graph(data: dict) -> GraphState:
    """Reconstruct GraphState from a stored dict."""
    nodes = {}
    for node_id, node_data in data.get("nodes", {}).items():
        tool_call = None
        if node_data.get("tool_call"):
            tool_call = ToolCall(**node_data["tool_call"])

        nodes[node_id] = NodeState(
            node_id=node_data["node_id"],
            task_id=node_data["task_id"],
            code=node_data["code"],
            console_output=node_data["console_output"],
            tool_output=node_data.get("tool_output"),
            tool_call=tool_call,
            edge=node_data.get("edge"),
            timestamp=node_data.get("timestamp", ""),
            request_index=node_data.get("request_index", 0),
        )

    graph = GraphState(
        session_id=data["session_id"],
        original_request=data["original_request"],
        requests=data.get("requests", [data["original_request"]]),
        current_task=data.get("current_task", "a"),
        current_step=data.get("current_step", 0),
        active_node=data.get("active_node"),
        active_edge=data.get("active_edge"),
        nodes=nodes,
    )

    return graph
