import sys
sys.path.insert(0, 'src')

from rdpge import Agent, tool, AgentResult

print("1. Imports OK")

# Test docstring description
fn = lambda path: "test"
fn.__doc__ = "Read file from disk."
t = tool()(fn) 
print(f"2. Docstring: {t.description}")

# Test explicit description  
t2 = tool("Explicit desc")(fn)
print(f"3. Explicit: {t2.description}")

# Test export_graph method exists
print(f"4. export_graph exists: {hasattr(AgentResult, 'export_graph')}")

# Test prompt validation 
agent_cls = Agent.__init__.__code__  # just checking the class is importable
print("5. Agent class OK")

print("ALL OK")
