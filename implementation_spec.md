# RDPGE-1 Implementation Specification

**Version:** 1.0  
**Date:** February 2026  
**Stack:** FastAPI, HTTPX, Asyncio, Docker, Kubernetes

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Structures](#2-data-structures)
3. [Component 1: Context Constructor](#3-component-1-context-constructor)
4. [Component 2: Execution Engine](#4-component-2-execution-engine)
5. [Component 3: Graph State Manager](#5-component-3-graph-state-manager)
6. [Component 4: Code Executor (Sandbox)](#6-component-4-code-executor-sandbox)
7. [Component 5: Tool Router](#7-component-5-tool-router)
8. [Main Orchestrator Loop](#8-main-orchestrator-loop)
9. [API Design (FastAPI)](#9-api-design-fastapi)
10. [Docker & Kubernetes Considerations](#10-docker--kubernetes-considerations)

---

## 1. System Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RDPGE-1 Agent System                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   FastAPI    │───▶│ Orchestrator │───▶│  Context Constructor │  │
│  │   Endpoint   │    │    Loop      │    │                      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                             │                       │               │
│                             ▼                       ▼               │
│                      ┌──────────────┐    ┌──────────────────────┐  │
│                      │   HTTPX      │◀───│   Graph State        │  │
│                      │ (LLM Client) │    │   Manager            │  │
│                      └──────────────┘    └──────────────────────┘  │
│                             │                       │               │
│                             ▼                       │               │
│                      ┌──────────────┐               │               │
│                      │    Code      │───────────────┘               │
│                      │   Executor   │                               │
│                      └──────────────┘                               │
│                             │                                        │
│                             ▼                                        │
│                      ┌──────────────┐                               │
│                      │    Tool      │                               │
│                      │   Router     │                               │
│                      └──────────────┘                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Summary

| Component | Responsibility |
|-----------|----------------|
| **Context Constructor** | Builds the `messages` list for LLM inference |
| **Execution Engine** | Orchestrates the inference → execute → update loop |
| **Graph State Manager** | Tracks nodes, edges, applies blurring rules |
| **Code Executor** | Sandboxed Python execution, extracts `action` dict |
| **Tool Router** | Dispatches tool calls to appropriate handlers |

---

## 2. Data Structures

### 2.1 Node State

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

class NodeStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    RESTORED = "restored"  # Via edge

@dataclass
class NodeState:
    """Represents a single execution node"""
    node_id: str                      # e.g., "node-a3"
    task_id: str                      # e.g., "a"
    step_num: int                     # e.g., 3
    status: NodeStatus
    
    # Content
    code: str                         # LLM-generated Python code
    console_output: str               # Captured stdout
    tool_output: Optional[str]        # Tool result (can be None or blurred)
    
    # Metadata
    tool_call: Optional[Dict[str, Any]]  # The tool_call from action dict
    edge: Optional[str]               # Edge declared by this node
    timestamp: str                    # ISO format
```

### 2.2 Graph State

```python
@dataclass
class GraphState:
    """Represents the entire execution graph"""
    session_id: str
    nodes: Dict[str, NodeState] = field(default_factory=dict)
    active_node: Optional[str] = None
    current_task: str = "a"           # Current task letter
    current_step: int = 0             # Step within current task
    
    # Edge tracking
    active_edges: List[str] = field(default_factory=list)  # Nodes currently restored
    
    def get_node(self, node_id: str) -> Optional[NodeState]:
        return self.nodes.get(node_id)
    
    def get_inactive_nodes(self) -> List[str]:
        return [
            node_id for node_id, node in self.nodes.items()
            if node.status == NodeStatus.INACTIVE
        ]
    
    def generate_next_node_id(self, new_task: bool = False) -> str:
        if new_task:
            # Move to next task letter
            self.current_task = chr(ord(self.current_task) + 1)
            self.current_step = 1
        else:
            self.current_step += 1
        
        return f"node-{self.current_task}{self.current_step}"
```

### 2.3 Action Dictionary Schema

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any]

class ActionDict(BaseModel):
    """Schema for the action dictionary output by LLM"""
    node: str = Field(..., description="Current execution node ID")
    edge: Optional[str] = Field(None, description="Edge to restore inactive node")
    tool_call: Optional[ToolCall] = Field(None, description="Tool to execute")
```

### 2.4 Execution Result

```python
@dataclass
class ExecutionResult:
    """Result from executing LLM-generated code"""
    success: bool
    action: Optional[ActionDict]
    code: str                         # Original code
    console_output: str
    error: Optional[str] = None
```

### 2.5 Message Format

```python
@dataclass
class Message:
    """Single message in the conversation"""
    role: str  # "system", "user", "assistant"
    content: str
```

---

## 3. Component 1: Context Constructor

### 3.1 Responsibility

Build the `messages` list for each LLM inference call. This includes:
- System prompt with Graph Manifest
- Previous node contexts (with blurring applied)
- Current turn prompt

### 3.2 Interface

```python
class ContextConstructor:
    """Builds messages list for LLM inference"""
    
    def __init__(self, system_prompt_template: str, codebase_snapshot: str):
        self.system_prompt_template = system_prompt_template
        self.codebase_snapshot = codebase_snapshot
    
    async def build_initial_context(
        self,
        user_request: str,
        graph_state: GraphState
    ) -> List[Message]:
        """Build context for the first turn (step 0)"""
        ...
    
    async def build_turn_context(
        self,
        graph_state: GraphState,
        latest_execution: ExecutionResult,
        tool_output: Optional[str]
    ) -> List[Message]:
        """Build context for subsequent turns"""
        ...
    
    def _build_graph_manifest(self, graph_state: GraphState) -> str:
        """Generate the Graph Manifest section"""
        ...
    
    def _build_node_context(
        self,
        node: NodeState,
        include_tool_output: bool
    ) -> str:
        """Build context string for a single node"""
        ...
```

### 3.3 System Prompt Template

```python
SYSTEM_PROMPT_TEMPLATE = '''You are an RDPGE-1 agent. You execute tasks by outputting Python code.

## RULES
1. Your output MUST be ONLY Python code wrapped in triple backticks with "python" language tag
2. Your code MUST define a variable named `action` which is a dictionary
3. Use comments for reasoning (this is your scratchpad)
4. The `action` dict must have: "node", "edge", "tool_call"

## ACTION SCHEMA
```python
action = {
    "node": "node-XX",        # Current node ID (you decide)
    "edge": "node-YY" | None, # Edge to restore context from inactive node
    "tool_call": {            # Optional - omit if no tool needed
        "name": "tool_name",
        "args": {...}
    } | None
}
```

## AVAILABLE TOOLS
- read_file(path: str, start_line: int = None, end_line: int = None)
- write_file(path: str, content: str)
- list_dir(path: str)
- run_command(command: str)
- search_code(query: str, path: str = ".")

## NODE NAMING
- Format: node-{task}{step} (e.g., node-a1, node-b3)
- New task = new letter (a, b, c, ...)
- Same task = increment step (1, 2, 3, ...)

{graph_manifest}

{codebase_snapshot}
'''
```

### 3.4 Graph Manifest Format

```python
def _build_graph_manifest(self, graph_state: GraphState) -> str:
    inactive_nodes = graph_state.get_inactive_nodes()
    
    manifest = f'''## RDPGE GRAPH MANIFEST
━━━━━━━━━━━━━━━━━━━━━━━
Active Node: {graph_state.active_node or "(awaiting first action)"}
Inactive Nodes: {inactive_nodes if inactive_nodes else "(none yet)"}

You may create an edge to any inactive node by setting "edge": "node-xxx"
This will restore that node's full context (including tool outputs).
'''
    return manifest
```

### 3.5 Node Context Builder

```python
def _build_node_context(
    self,
    node: NodeState,
    include_tool_output: bool
) -> str:
    """Build context string for a single node"""
    
    context = f'''=== [{node.node_id}] ===

## CODE:
```python
{node.code}
```

## CONSOLE OUTPUT:
{node.console_output or "(none)"}
'''
    
    if include_tool_output and node.tool_output:
        context += f'''
## TOOL RESULT:
{node.tool_output}
'''
    elif not include_tool_output:
        context += '''
## TOOL RESULT:
(blurred - create edge to this node to restore)
'''
    
    context += f'''=== END [{node.node_id}] ===
'''
    return context
```

### 3.6 Full Context Build (Subsequent Turns)

```python
async def build_turn_context(
    self,
    graph_state: GraphState,
    latest_execution: ExecutionResult,
    tool_output: Optional[str]
) -> List[Message]:
    """Build context for subsequent turns"""
    
    messages = []
    
    # 1. System prompt with updated manifest
    system_content = self.system_prompt_template.format(
        graph_manifest=self._build_graph_manifest(graph_state),
        codebase_snapshot=self.codebase_snapshot
    )
    messages.append(Message(role="system", content=system_content))
    
    # 2. Original user request
    messages.append(Message(role="user", content=graph_state.original_request))
    
    # 3. All previous nodes (with blurring)
    for node_id, node in graph_state.nodes.items():
        # Determine if tool output should be included
        include_tool_output = (
            node.status == NodeStatus.ACTIVE or
            node.status == NodeStatus.RESTORED or
            node_id in graph_state.active_edges
        )
        
        node_context = self._build_node_context(node, include_tool_output)
        messages.append(Message(role="assistant", content=node_context))
    
    # 4. Latest execution result (current turn)
    current_context = f'''=== EXECUTION RESULT FOR [{graph_state.active_node}] ===

## YOUR CODE:
```python
{latest_execution.code}
```

## CONSOLE OUTPUT:
{latest_execution.console_output or "(none)"}

## TOOL RESULT:
{tool_output or "(no tool called)"}

=== END EXECUTION RESULT ===

Continue with your task. Output your next action as Python code.
'''
    messages.append(Message(role="user", content=current_context))
    
    return messages
```

---

## 4. Component 2: Execution Engine

### 4.1 Responsibility

Orchestrate the main agent loop:
1. Get LLM inference
2. Parse code from response
3. Execute code
4. Run tool if specified
5. Update graph state
6. Loop until task complete

### 4.2 Interface

```python
class ExecutionEngine:
    """Main orchestrator for RDPGE-1 agent execution"""
    
    def __init__(
        self,
        llm_client: "LLMClient",
        code_executor: "CodeExecutor",
        tool_router: "ToolRouter",
        context_constructor: "ContextConstructor",
        graph_manager: "GraphStateManager"
    ):
        self.llm_client = llm_client
        self.code_executor = code_executor
        self.tool_router = tool_router
        self.context_constructor = context_constructor
        self.graph_manager = graph_manager
    
    async def run(
        self,
        user_request: str,
        session_id: str,
        max_steps: int = 50
    ) -> "AgentResult":
        """Run the agent until completion or max steps"""
        ...
    
    async def execute_step(
        self,
        graph_state: GraphState,
        messages: List[Message]
    ) -> Tuple[ExecutionResult, Optional[str]]:
        """Execute a single step: inference → code exec → tool"""
        ...
```

### 4.3 Main Loop Implementation

```python
async def run(
    self,
    user_request: str,
    session_id: str,
    max_steps: int = 50
) -> "AgentResult":
    """Run the agent until completion or max steps"""
    
    # Initialize graph state
    graph_state = GraphState(session_id=session_id)
    graph_state.original_request = user_request
    
    # Step counter
    step = 0
    
    # Build initial context
    messages = await self.context_constructor.build_initial_context(
        user_request=user_request,
        graph_state=graph_state
    )
    
    while step < max_steps:
        step += 1
        print(f"[Step {step}] Executing...")
        
        # 1. Get LLM inference
        llm_response = await self.llm_client.inference(messages)
        
        # 2. Parse Python code from response
        code = self._extract_code(llm_response)
        if not code:
            return AgentResult(
                success=False,
                error="LLM did not return valid Python code"
            )
        
        # 3. Execute code in sandbox
        exec_result = await self.code_executor.execute(code)
        if not exec_result.success:
            # Could retry or handle error
            return AgentResult(
                success=False,
                error=f"Code execution failed: {exec_result.error}"
            )
        
        # 4. Check if task is complete (no tool call, or special signal)
        if exec_result.action.tool_call is None:
            # Agent signaled completion
            return AgentResult(
                success=True,
                graph_state=graph_state,
                final_node=exec_result.action.node
            )
        
        # 5. Execute tool
        tool_output = await self.tool_router.execute(
            exec_result.action.tool_call
        )
        
        # 6. Update graph state
        await self.graph_manager.update(
            graph_state=graph_state,
            action=exec_result.action,
            code=code,
            console_output=exec_result.console_output,
            tool_output=tool_output
        )
        
        # 7. Build next context
        messages = await self.context_constructor.build_turn_context(
            graph_state=graph_state,
            latest_execution=exec_result,
            tool_output=tool_output
        )
    
    return AgentResult(
        success=False,
        error=f"Max steps ({max_steps}) exceeded"
    )
```

### 4.4 Code Extraction

```python
import re

def _extract_code(self, llm_response: str) -> Optional[str]:
    """Extract Python code from LLM response"""
    
    # Pattern: ```python ... ```
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, llm_response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    return None
```

---

## 5. Component 3: Graph State Manager

### 5.1 Responsibility

- Track all nodes and their states
- Apply blurring rules
- Handle edge restoration
- Update node statuses

### 5.2 Interface

```python
class GraphStateManager:
    """Manages the execution graph state"""
    
    async def update(
        self,
        graph_state: GraphState,
        action: ActionDict,
        code: str,
        console_output: str,
        tool_output: Optional[str]
    ) -> None:
        """Update graph state after a step execution"""
        ...
    
    def apply_edge(
        self,
        graph_state: GraphState,
        target_node_id: str
    ) -> None:
        """Restore context for a node via edge"""
        ...
    
    def apply_blurring(
        self,
        graph_state: GraphState
    ) -> None:
        """Apply blurring rules to all nodes"""
        ...
```

### 5.3 Update Implementation

```python
async def update(
    self,
    graph_state: GraphState,
    action: ActionDict,
    code: str,
    console_output: str,
    tool_output: Optional[str]
) -> None:
    """Update graph state after a step execution"""
    
    node_id = action.node
    
    # 1. Mark previous active node as inactive
    if graph_state.active_node:
        prev_node = graph_state.get_node(graph_state.active_node)
        if prev_node:
            prev_node.status = NodeStatus.INACTIVE
    
    # 2. Clear previous edge restorations
    for restored_id in graph_state.active_edges:
        restored_node = graph_state.get_node(restored_id)
        if restored_node:
            restored_node.status = NodeStatus.INACTIVE
    graph_state.active_edges.clear()
    
    # 3. Handle new edge if declared
    if action.edge:
        self.apply_edge(graph_state, action.edge)
    
    # 4. Parse node ID to get task/step
    task_id, step_num = self._parse_node_id(node_id)
    
    # 5. Create or update current node
    node = NodeState(
        node_id=node_id,
        task_id=task_id,
        step_num=step_num,
        status=NodeStatus.ACTIVE,
        code=code,
        console_output=console_output,
        tool_output=tool_output,
        tool_call=action.tool_call.dict() if action.tool_call else None,
        edge=action.edge,
        timestamp=datetime.now().isoformat()
    )
    
    graph_state.nodes[node_id] = node
    graph_state.active_node = node_id
    graph_state.current_task = task_id
    graph_state.current_step = step_num

def _parse_node_id(self, node_id: str) -> Tuple[str, int]:
    """Parse node-a3 into ('a', 3)"""
    # Pattern: node-{letter}{number}
    match = re.match(r'node-([a-z])(\d+)', node_id)
    if match:
        return match.group(1), int(match.group(2))
    raise ValueError(f"Invalid node ID format: {node_id}")

def apply_edge(
    self,
    graph_state: GraphState,
    target_node_id: str
) -> None:
    """Restore context for a node via edge"""
    
    target_node = graph_state.get_node(target_node_id)
    if target_node:
        target_node.status = NodeStatus.RESTORED
        graph_state.active_edges.append(target_node_id)
```

---

## 6. Component 4: Code Executor (Sandbox)

### 6.1 Responsibility

- Execute LLM-generated Python code safely
- Extract `action` dictionary from local namespace
- Capture console output
- Handle errors gracefully

### 6.2 Interface

```python
class CodeExecutor:
    """Sandboxed Python code executor"""
    
    def __init__(self, allowed_imports: List[str] = None):
        self.allowed_imports = allowed_imports or []
    
    async def execute(self, code: str) -> ExecutionResult:
        """Execute code and extract action dictionary"""
        ...
```

### 6.3 Implementation

```python
import io
import sys
from typing import Dict, Any

class CodeExecutor:
    """Sandboxed Python code executor"""
    
    def __init__(self, allowed_imports: List[str] = None):
        self.allowed_imports = allowed_imports or []
    
    async def execute(self, code: str) -> ExecutionResult:
        """Execute code and extract action dictionary"""
        
        # Create restricted builtins
        safe_builtins = {
            "print": print,
            "len": len,
            "range": range,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "True": True,
            "False": False,
            "None": None,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "isinstance": isinstance,
            "type": type,
            # Add more as needed
        }
        
        global_namespace = {"__builtins__": safe_builtins}
        local_namespace = {}
        
        # Capture stdout
        console_buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = console_buffer
        
        try:
            # Execute the code
            exec(code, global_namespace, local_namespace)
            
            # Restore stdout
            sys.stdout = old_stdout
            console_output = console_buffer.getvalue()
            
            # Extract action
            if 'action' not in local_namespace:
                return ExecutionResult(
                    success=False,
                    action=None,
                    code=code,
                    console_output=console_output,
                    error="Code did not define 'action' variable"
                )
            
            action_raw = local_namespace['action']
            
            # Validate action structure
            if not isinstance(action_raw, dict):
                return ExecutionResult(
                    success=False,
                    action=None,
                    code=code,
                    console_output=console_output,
                    error=f"'action' must be dict, got {type(action_raw)}"
                )
            
            # Parse into ActionDict
            try:
                action = ActionDict(**action_raw)
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    action=None,
                    code=code,
                    console_output=console_output,
                    error=f"Invalid action structure: {e}"
                )
            
            return ExecutionResult(
                success=True,
                action=action,
                code=code,
                console_output=console_output
            )
            
        except Exception as e:
            sys.stdout = old_stdout
            console_output = console_buffer.getvalue()
            
            return ExecutionResult(
                success=False,
                action=None,
                code=code,
                console_output=console_output,
                error=str(e)
            )
```

---

## 7. Component 5: Tool Router

### 7.1 Responsibility

- Dispatch tool calls to appropriate handlers
- Execute file operations, commands, etc.
- Return tool outputs

### 7.2 Interface

```python
class ToolRouter:
    """Routes tool calls to handlers"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.tools: Dict[str, Callable] = {}
        self._register_default_tools()
    
    async def execute(self, tool_call: ToolCall) -> str:
        """Execute a tool and return output"""
        ...
    
    def register_tool(self, name: str, handler: Callable):
        """Register a custom tool"""
        self.tools[name] = handler
```

### 7.3 Default Tools

```python
def _register_default_tools(self):
    self.tools = {
        "read_file": self._read_file,
        "write_file": self._write_file,
        "list_dir": self._list_dir,
        "run_command": self._run_command,
        "search_code": self._search_code,
    }

async def _read_file(
    self,
    path: str,
    start_line: int = None,
    end_line: int = None
) -> str:
    """Read file contents, optionally specific lines"""
    full_path = os.path.join(self.workspace_path, path)
    
    async with aiofiles.open(full_path, 'r') as f:
        lines = await f.readlines()
    
    if start_line is not None and end_line is not None:
        # 1-indexed, inclusive
        lines = lines[start_line-1:end_line]
    
    return ''.join(lines)

async def _write_file(self, path: str, content: str) -> str:
    """Write content to file"""
    full_path = os.path.join(self.workspace_path, path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    async with aiofiles.open(full_path, 'w') as f:
        await f.write(content)
    
    return f"Successfully wrote {len(content)} bytes to {path}"

async def _list_dir(self, path: str) -> str:
    """List directory contents with file sizes"""
    full_path = os.path.join(self.workspace_path, path)
    
    result = []
    for item in os.listdir(full_path):
        item_path = os.path.join(full_path, item)
        if os.path.isdir(item_path):
            result.append(f"📁 {item}/")
        else:
            size = os.path.getsize(item_path)
            lines = sum(1 for _ in open(item_path, 'rb'))
            result.append(f"📄 {item} ({lines} lines)")
    
    return '\n'.join(result)
```

---

## 8. Main Orchestrator Loop

### 8.1 Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RDPGE-1 Execution Flow                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  START                                                               │
│    │                                                                 │
│    ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  STEP 0: Initialize                                          │    │
│  │  - Create GraphState                                         │    │
│  │  - Build initial messages (system + manifest + user request) │    │
│  └─────────────────────────────────────────────────────────────┘    │
│    │                                                                 │
│    ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  STEP 1+: Execution Loop                                     │    │
│  │  ┌─────────────────────────────────────────────────────────┐ │    │
│  │  │ 1. Call LLM with messages                               │ │    │
│  │  │ 2. Extract Python code from response                    │ │    │
│  │  │ 3. Execute code in sandbox                              │ │    │
│  │  │ 4. Extract 'action' dictionary                          │ │    │
│  │  │ 5. If tool_call exists, execute tool                    │ │    │
│  │  │ 6. Update GraphState (node, edge, blurring)             │ │    │
│  │  │ 7. Build next messages context                          │ │    │
│  │  │ 8. Check termination condition                          │ │    │
│  │  └─────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
│    │                                                                 │
│    ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  COMPLETE: Return AgentResult with final graph              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Step Counter Logic

```python
# The step counter increments AFTER each successful execution
# Step 0 = Initial context (before first LLM call)
# Step 1 = First LLM response processed
# Step N = Nth LLM response processed

# Context at each step includes:
# - All previous nodes (blurred unless edge restored)
# - Current execution result (code + console + tool output)
```

---

## 9. API Design (FastAPI)

### 9.1 Endpoints

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="RDPGE-1 Agent API")

class AgentRequest(BaseModel):
    user_request: str
    session_id: Optional[str] = None
    max_steps: int = 50

class AgentResponse(BaseModel):
    session_id: str
    status: str  # "running", "completed", "failed"
    current_step: int
    active_node: Optional[str]
    message: Optional[str]

@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """Start a new agent session"""
    ...

@app.get("/agent/{session_id}/status", response_model=AgentResponse)
async def get_status(session_id: str):
    """Get status of running agent"""
    ...

@app.get("/agent/{session_id}/graph")
async def get_graph(session_id: str):
    """Get the execution graph for visualization"""
    ...

@app.post("/agent/{session_id}/stop")
async def stop_agent(session_id: str):
    """Stop a running agent"""
    ...
```

### 9.2 WebSocket for Streaming

```python
from fastapi import WebSocket

@app.websocket("/agent/{session_id}/stream")
async def stream_agent(websocket: WebSocket, session_id: str):
    """Stream agent execution in real-time"""
    await websocket.accept()
    
    # Subscribe to agent events
    async for event in agent_event_stream(session_id):
        await websocket.send_json({
            "type": event.type,  # "step", "tool_call", "complete", "error"
            "data": event.data
        })
```

---

## 10. Docker & Kubernetes Considerations

### 10.1 Docker Setup

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 10.2 Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLM_API_KEY=${LLM_API_KEY}
      - LLM_API_URL=${LLM_API_URL}
    volumes:
      - ./workspace:/app/workspace  # Agent workspace
      - ./logs:/app/logs            # Execution logs
    
  # Optional: Redis for session persistence
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### 10.3 Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rdpge1-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rdpge1-agent
  template:
    metadata:
      labels:
        app: rdpge1-agent
    spec:
      containers:
      - name: agent
        image: rdpge1-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: rdpge1-agent-service
spec:
  selector:
    app: rdpge1-agent
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Summary: File Structure

```
rdpge1/
├── main.py                    # FastAPI app entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── k8s/
│   └── deployment.yaml
│
├── core/
│   ├── __init__.py
│   ├── models.py              # Data structures (NodeState, GraphState, etc.)
│   ├── context.py             # ContextConstructor
│   ├── executor.py            # CodeExecutor (sandbox)
│   ├── engine.py              # ExecutionEngine (main loop)
│   ├── graph.py               # GraphStateManager
│   └── tools.py               # ToolRouter
│
├── llm/
│   ├── __init__.py
│   ├── client.py              # HTTPX-based LLM client
│   └── providers/
│       ├── openai.py
│       ├── anthropic.py
│       └── qwen.py
│
├── api/
│   ├── __init__.py
│   ├── routes.py              # FastAPI routes
│   └── websocket.py           # WebSocket streaming
│
└── config/
    ├── __init__.py
    └── settings.py            # Environment config
```

---

## Next Steps

1. **Start with `core/models.py`** - Define all data structures
2. **Implement `core/executor.py`** - The sandbox is critical
3. **Implement `core/context.py`** - Message construction
4. **Implement `core/graph.py`** - State management
5. **Implement `core/engine.py`** - Main loop
6. **Add `llm/client.py`** - HTTPX async client
7. **Build `api/routes.py`** - FastAPI endpoints
8. **Test locally with Docker Compose**
9. **Deploy to Kubernetes**

Good luck with your implementation! 🚀
