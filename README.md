# RDPGE

**Runtime Dynamic & Probabilistic Graph Execution**

A novel agentic AI framework that treats LLM outputs as probabilistic events, not deterministic programs.

RDPGE replaces the flat ReAct loop with a dynamic execution graph — giving agents structured memory, selective context, and real-time situational awareness.

---

## Why RDPGE?

Standard agentic frameworks (ReAct, function-calling loops) suffer from three core problems:

| Problem | What Happens | RDPGE's Solution |
|---------|-------------|-----------------|
| **Myopia** | The agent only sees recent context. Work done 15 steps ago fades from attention. | **Graph Manifest** — a live dashboard showing all tasks, distances, references, and step budget. Updated every turn. |
| **Greedy execution** | The agent does the first thing that seems useful without considering the bigger picture. | **Task-based organization** — work is structured into named tasks. The agent sees the full map before acting. |
| **Context waste** | All tool outputs stay in context forever, consuming tokens even when irrelevant. | **Context blurring** — inactive tasks' tool outputs are replaced with `[BLURRED]`. Restored on demand via edges. |

## Key Ideas

- **Code-as-interface** — The LLM outputs Python code, not JSON tool calls. Comments serve as chain-of-thought. An `action` dictionary is the structured interface.
- **Probabilistic graph** — Execution is tracked as a dynamic graph of nodes organized by tasks. The LLM creates tasks and nodes as needed.
- **Context blurring** — Only the active task's tool outputs are visible. Everything else is `[BLURRED]`. The LLM can restore any task's context by creating an edge.
- **Graph manifest** — Every turn, the LLM sees a live dashboard: active node, current task, step budget, task distances, and inter-task references.
- **Signal tools** — Built-in tools (`complete`, `ask_user`, `surrender`) that control execution flow explicitly.

## Install

```bash
pip install rdpge
```

## Quick Start

```python
import asyncio
from rdpge import Agent, tool
from rdpge.llm import OpenAIAdapter, LLMConfig

# 1. Define tools
@tool()
def read_file(path: str) -> str:
    """Read file contents from disk."""
    with open(path) as f:
        return f.read()

@tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file on disk."""
    with open(path, "w") as f:
        f.write(content)
    return f"Written {len(content)} bytes to {path}"

# 2. Create agent
agent = Agent(
    llm=OpenAIAdapter(LLMConfig(
        model="gpt-4o",
        api_key="sk-...",
    )),
    tools=[read_file, write_file],
    instructions="You are a code assistant.",
    max_steps=25,
)

# 3. Run
async def main():
    result = await agent.run("Read auth.py and fix the login bug")
    print(f"Status: {result.status}")
    print(f"Steps used: {result.steps}")
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

## Using with Together AI

```python
from rdpge.llm import OpenAIAdapter, LLMConfig

llm = OpenAIAdapter(LLMConfig(
    model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    api_key="your-together-key",
    base_url="https://api.together.xyz/v1",
))
agent = Agent(llm=llm, tools=[...])
```

## Using with Anthropic

```python
from rdpge.llm import AnthropicAdapter, LLMConfig

llm = AnthropicAdapter(LLMConfig(
    model="claude-sonnet-4-20250514",
    api_key="sk-ant-...",
))
agent = Agent(llm=llm, tools=[...])
```

## How It Works

```
User Request
     │
     ▼
┌─────────────────────────────────────────────┐
│  EXECUTION LOOP                             │
│                                             │
│  1. Build Context                           │
│     ├── System Prompt                       │
│     ├── Graph Manifest (live dashboard)     │
│     └── Node History (with blurring)        │
│                                             │
│  2. LLM Generates Python Code               │
│     └── action = {node, reason, tool_call}  │
│                                             │
│  3. Execute Code in Sandbox                 │
│     └── Extract action dict                 │
│                                             │
│  4. Route Tool Call                         │
│     ├── Signal tool? → Handle internally    │
│     └── Regular tool? → Execute via registry│
│                                             │
│  5. Update Graph                            │
│     ├── Record node in task                 │
│     ├── Apply blurring to inactive tasks    │
│     └── Save state (multi-turn)             │
│                                             │
│  6. Loop until: complete / surrender /      │
│     ask_user / max_steps                    │
└─────────────────────────────────────────────┘
     │
     ▼
  AgentResult(status, reason, steps, trace, graph)
```

### The Graph Manifest

Every turn, the LLM sees this dashboard in its context:

```
Active Node: node-b2
Current Task: b
Step: 6 of 25
Active Edge: (none)

Task Map:
  Task A: 2 steps ago | 0 references | 3 steps
  Task B: active | 0 references | 2 steps
```

- **Distance** tells the LLM how far back a task is (recency).
- **References** show inter-task dependencies (importance).
- **Step counter** shows remaining budget (the hard execution limit).

### Context Blurring

When the LLM switches from Task A to Task B:

```
# Task A's tool outputs become:
Tool: [BLURRED]

# Task B's tool outputs remain fully visible:
Tool: def login(user, pwd): ...
```

The LLM can restore Task A's context by setting `edge: "node-a"` in its action.

## Signal Tools

Built-in tools that control execution flow:

| Signal | Args | Effect |
|--------|------|--------|
| `complete` | `{}` | Task is done. Loop ends. |
| `ask_user` | `{"question": str}` | Pauses execution. Returns the question to the caller. |
| `surrender` | `{"reason": str}` | Cannot accomplish the task. Loop ends. |

Developer-side abort:
```python
# From a UI button or hook callback:
agent.abort("User pressed cancel")
```

## Multi-Turn Sessions

RDPGE preserves full graph state between conversations:

```python
# Turn 1
result = await agent.run("Read the codebase and find bugs")
# result.status == "awaiting_input" (agent used ask_user)

# Turn 2 — agent remembers everything from Turn 1
result = await agent.run("Focus on auth.py")
```

### Persistence

```python
from rdpge import Agent, InMemoryStore

# In-memory (default)
agent = Agent(llm=llm, tools=[...], store=InMemoryStore())

# Custom store (Redis, database, etc.)
# Implement the SessionStore protocol:
class RedisStore:
    async def save(self, session_id: str, data: dict) -> None: ...
    async def load(self, session_id: str) -> dict | None: ...
    async def delete(self, session_id: str) -> None: ...
    async def list_sessions(self) -> list[str]: ...
```

## Event Hooks

Monitor and react to agent behavior:

```python
@agent.on("step_end")
async def log_step(data):
    print(f"Step {data['step']}: {data['node_id']} → {data['tool']}")

@agent.on("complete")
async def on_done(data):
    print(f"Finished in {data['steps']} steps")

@agent.on("surrender")
async def on_surrender(data):
    print(f"Gave up: {data['reason']}")
```

Available events: `step_start`, `step_end`, `tool_called`, `complete`, `ask_user`, `surrender`, `abort`, `error`

## Execution Traces

Every run produces a detailed trace:

```python
result = await agent.run("Fix the bug")
summary = result.trace.summary()

print(summary)
# {
#   "session_id": "a1b2c3d4",
#   "total_steps": 5,
#   "total_duration_ms": 12340,
#   "tools_used": {"read_file": 2, "write_file": 1, "complete": 1, ...},
#   "nodes": ["node-a1", "node-a2", "node-a3", "node-b1", "node-b2"],
# }
```

## API Reference

### `Agent`

```python
Agent(
    llm,                          # LLMProvider instance
    tools: list = None,           # List of @tool() or BaseTool instances
    instructions: str = "",       # Domain-specific instructions for the LLM
    instructions_file: str = None,# Or load instructions from a file
    max_steps: int = 25,          # Hard execution limit
    store = None,                 # SessionStore for persistence
    signal_handlers: dict = None, # Override signal tool behavior
)
```

**Methods:**
- `await agent.run(request)` → `AgentResult`
- `agent.abort(reason)` — stop execution from outside
- `agent.new_session()` — start fresh
- `await agent.load_session(session_id)` → `bool`
- `agent.on(event)` — decorator for event hooks

### `AgentResult`

```python
@dataclass
class AgentResult:
    success: bool           # True if task completed successfully
    status: str             # "completed" | "awaiting_input" | "surrendered" | "aborted" | "max_steps" | "error"
    reason: str             # Human-readable summary
    steps: int              # Number of steps executed
    session_id: str         # Session identifier
    trace: SessionTrace     # Execution trace
    graph: GraphState       # Final graph state
    error: str = ""         # Error details (if status == "error")
```

### `@tool()`

```python
@tool()
def my_tool(param: str) -> str:
    """Tool description (used in the LLM prompt)."""
    return result

# Or with explicit description:
@tool("Override the docstring description")
def my_tool(param: str) -> str:
    ...
```

## Architecture

RDPGE is built on four principles:

1. **Respect probabilistic nature** — LLM outputs are unpredictable. The framework handles malformed outputs, retries, and edge cases structurally.

2. **Overcome myopia** — The graph manifest gives the LLM a bird's-eye view of its entire session, preventing short-sighted decisions.

3. **Ensure task accuracy** — Signal tools (`surrender`, `ask_user`) let the agent be honest about its limitations instead of producing garbage output.

4. **Complete observability** — Traces, hooks, and graph export provide full transparency into agent behavior at every level.

## License

MIT

## Author

**Jahanzeb Ahmed** — [GitHub](https://github.com/Jahanzeb-git)
