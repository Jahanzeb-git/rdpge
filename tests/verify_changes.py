import sys
import asyncio
sys.path.insert(0, 'src')

from rdpge import Agent, tool, AgentResult, InMemoryStore, SessionStore, GraphState, NodeState
from rdpge.storage.base import serialize_graph, deserialize_graph
from rdpge.core.models import ToolCall

print("=" * 60)
print("RDPGE Multi-Turn Persistence — Verification")
print("=" * 60)

# ---- Test 1: Imports ----
print("\n1. Imports OK ✓")

# ---- Test 2: InMemoryStore basics ----
async def test_store():
    store = InMemoryStore()
    
    # save + load
    await store.save("sess1", {"key": "value1"})
    await store.save("sess2", {"key": "value2"})
    
    data = await store.load("sess1")
    assert data == {"key": "value1"}, f"Expected value1, got {data}"
    
    # list
    sessions = await store.list_sessions()
    assert set(sessions) == {"sess1", "sess2"}, f"Expected sess1,sess2, got {sessions}"
    
    # load non-existent
    missing = await store.load("nonexistent")
    assert missing is None, f"Expected None, got {missing}"
    
    # delete
    await store.delete("sess1")
    sessions = await store.list_sessions()
    assert sessions == ["sess2"], f"Expected [sess2], got {sessions}"
    
    # delete non-existent (should not error)
    await store.delete("nonexistent")
    
    print("2. InMemoryStore basics OK ✓")

asyncio.run(test_store())

# ---- Test 3: SessionStore protocol check ----
assert isinstance(InMemoryStore(), SessionStore), "InMemoryStore should satisfy SessionStore protocol"
print("3. SessionStore protocol OK ✓")

# ---- Test 4: GraphState serialization round-trip ----
graph = GraphState(
    session_id="test123",
    original_request="Fix the bug",
    requests=["Fix the bug", "Now add tests"],
    current_task="b",
    current_step=3,
    active_node="node-b3",
    active_edge="node-a",
)

# Add nodes with request_index
graph.nodes["node-a1"] = NodeState(
    node_id="node-a1",
    task_id="a",
    code="action = {...}",
    console_output="output here",
    tool_output="file contents",
    tool_call=ToolCall(name="read_file", args={"path": "auth.py"}),
    edge=None,
    request_index=0,
)
graph.nodes["node-b1"] = NodeState(
    node_id="node-b1",
    task_id="b",
    code="action = {...}",
    console_output="",
    tool_output="test added",
    tool_call=ToolCall(name="write_file", args={"path": "test.py", "content": "..."}),
    edge="node-a",
    request_index=1,
)

# Serialize
serialized = serialize_graph(graph)
assert isinstance(serialized, dict), "serialize_graph should return dict"
assert serialized["session_id"] == "test123"
assert serialized["requests"] == ["Fix the bug", "Now add tests"]
assert len(serialized["nodes"]) == 2
assert serialized["nodes"]["node-a1"]["request_index"] == 0
assert serialized["nodes"]["node-b1"]["request_index"] == 1
assert serialized["nodes"]["node-a1"]["tool_call"]["name"] == "read_file"

# Deserialize
restored = deserialize_graph(serialized)
assert isinstance(restored, GraphState), "deserialize_graph should return GraphState"
assert restored.session_id == "test123"
assert restored.original_request == "Fix the bug"
assert restored.requests == ["Fix the bug", "Now add tests"]
assert restored.current_task == "b"
assert restored.current_step == 3
assert restored.active_node == "node-b3"
assert restored.active_edge == "node-a"
assert len(restored.nodes) == 2
assert restored.nodes["node-a1"].request_index == 0
assert restored.nodes["node-b1"].request_index == 1
assert restored.nodes["node-a1"].tool_call.name == "read_file"
assert restored.nodes["node-b1"].tool_call.args == {"path": "test.py", "content": "..."}
assert restored.nodes["node-b1"].edge == "node-a"

print("4. GraphState serialization round-trip OK ✓")

# ---- Test 5: Store + Serialization integration ----
async def test_store_integration():
    store = InMemoryStore()
    
    # Save a graph
    await store.save("test123", serialize_graph(graph))
    
    # Load it back
    data = await store.load("test123")
    restored = deserialize_graph(data)
    
    assert restored.session_id == "test123"
    assert len(restored.nodes) == 2
    assert restored.requests == ["Fix the bug", "Now add tests"]
    
    print("5. Store + Serialization integration OK ✓")

asyncio.run(test_store_integration())

# ---- Test 6: Agent API surface ----
class FakeLLM:
    async def generate(self, messages):
        return "no code"

agent = Agent(llm=FakeLLM(), tools=[])

# Check session management methods exist
assert hasattr(agent, 'session_id'), "Agent should have session_id property"
assert hasattr(agent, 'new_session'), "Agent should have new_session method"
assert hasattr(agent, 'load_session'), "Agent should have load_session method"
assert hasattr(agent, 'list_sessions'), "Agent should have list_sessions method"
assert hasattr(agent, 'store'), "Agent should have store attribute"
assert isinstance(agent.store, InMemoryStore), "Default store should be InMemoryStore"

# session_id starts as None
assert agent.session_id is None, f"Expected None, got {agent.session_id}"

# new_session should work without error
agent.new_session()
assert agent.session_id is None, "After new_session, session_id should be None"

print("6. Agent API surface OK ✓")

# ---- Test 7: Multi-turn context building ----
from rdpge.core.context import ContextConstructor
from rdpge.core.models import Message

# Minimal context builder (no template needed for this test)
# We'll test build_turn_context directly
graph2 = GraphState(
    session_id="multi1",
    original_request="Read auth.py",
    requests=["Read auth.py", "Fix the bug"],
    current_task="a",
    current_step=2,
)

# Add nodes for request 0
graph2.nodes["node-a1"] = NodeState(
    node_id="node-a1",
    task_id="a",
    code="action = {'node': 'node-a1', ...}",
    console_output="",
    tool_output="def login(): pass",
    request_index=0,
)

# Add nodes for request 1  
graph2.nodes["node-a2"] = NodeState(
    node_id="node-a2",
    task_id="a",
    code="action = {'node': 'node-a2', ...}",
    console_output="",
    tool_output="Fixed login()",
    request_index=1,
)

# Build context
ctx = ContextConstructor(
    domain_context="Test agent",
    tools_prompt="## AVAILABLE TOOLS\n(none)"
)
messages = ctx.build_turn_context(graph2)

# Verify structure: system + [USER TASK 1] + node-a1 + [USER TASK 2] + node-a2
# messages[0] = system
# messages[1] = [USER TASK] Read auth.py
# messages[2] = assistant (node-a1 code)
# messages[3] = user (node-a1 result)
# messages[4] = [USER TASK] Fix the bug
# messages[5] = assistant (node-a2 code)
# messages[6] = user (node-a2 result)

assert messages[0].role == "system", f"Expected system, got {messages[0].role}"
assert messages[1].role == "user" and "Read auth.py" in messages[1].content
assert messages[2].role == "assistant" and "node-a1" in messages[2].content
assert messages[3].role == "user" and "EXECUTION RESULT: node-a1" in messages[3].content
assert messages[4].role == "user" and "Fix the bug" in messages[4].content
assert messages[5].role == "assistant" and "node-a2" in messages[5].content
assert messages[6].role == "user" and "EXECUTION RESULT: node-a2" in messages[6].content

print("7. Multi-turn context building OK ✓")

# ---- Test 8: Engine multi-turn state ----
from rdpge.core.engine import ExecutionEngine

engine = ExecutionEngine(
    llm=FakeLLM(),
    registry=None,
    context_builder=ctx,
    hooks=None,
    store=InMemoryStore(),
)

assert engine.graph is None, "Engine should start with no graph"
assert engine.session_id is None, "Engine should start with no session_id"

# test reset
engine.graph = graph2
engine.session_id = "multi1"
engine.reset()
assert engine.graph is None, "After reset, graph should be None"
assert engine.session_id is None, "After reset, session_id should be None"

# test set_graph
engine.set_graph(graph2)
assert engine.graph is graph2, "set_graph should set the graph"
assert engine.session_id == "multi1", "set_graph should set session_id"

print("8. Engine multi-turn state OK ✓")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
