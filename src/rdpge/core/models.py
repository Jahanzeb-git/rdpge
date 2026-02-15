# models.py — All data structures for RDPGE
#
# Contains:
#   - Message         (role + content for LLM messages)
#   - NodeState       (single execution node in the graph)
#   - GraphState      (the entire execution graph)
#   - ActionDict      (parsed output from LLM's code)
#   - ToolCall        (tool name + args from action dict)
#   - ExecutionResult  (result from sandbox code execution)

from dataclasses import dataclass, field
from typing import Optional, Any
from pydantic import BaseModel
from datetime import datetime

# ------ LLM INTERFACE STRUCTURES ------
class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]

class ActionDict(BaseModel):
    node: str
    edge: Optional[str] = None
    reason: str
    tool_call: Optional[ToolCall] = None

# ------ INTERNAL STRUCTURES ------
@dataclass
class Message:
    role: str
    content: str

@dataclass
class ExecutionResult:
    success: bool
    action: Optional[ActionDict]
    code: str
    console_output: str
    error: Optional[str] = None

# ------ GRAPH STRUCTURES ------
@dataclass
class NodeState:
    node_id: str
    task_id: str
    code: str
    console_output: str
    tool_output: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    edge: Optional[str] = None
    request_index: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class GraphState:
    session_id: str
    original_request: str
    nodes: dict[str, NodeState] = field(default_factory=dict)
    requests: list[str] = field(default_factory=list)
    active_node: Optional[str] = None
    active_edge: Optional[str] = None
    current_task: str = "a"
    current_step: int = 0
    last_error: Optional[str] = None

    def get_node(self, node_id: str) -> Optional[NodeState]:
        return self.nodes.get(node_id)
