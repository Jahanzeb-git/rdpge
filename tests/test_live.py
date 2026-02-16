"""
First real integration test of RDPGE with a live LLM.
Uses Together AI's Qwen3 Coder 480B via the published PyPI package.
"""

import asyncio
from rdpge import Agent, tool
from rdpge.llm import OpenAIAdapter, LLMConfig

# ---- Define tools ----

@tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool()
def lookup_info(topic: str) -> str:
    """Look up information about a programming topic. Returns a brief explanation."""
    data = {
        "python": "Python is a high-level, interpreted programming language known for its readability.",
        "asyncio": "asyncio is Python's built-in library for writing concurrent code using async/await syntax.",
        "fastapi": "FastAPI is a modern Python web framework for building APIs with automatic OpenAPI docs.",
        "rust": "Rust is a systems programming language focused on safety, speed, and concurrency.",
    }
    key = topic.lower().strip()
    for k, v in data.items():
        if k in key:
            return v
    return f"No information found for: {topic}"


# ---- Create agent with debug wrapper ----

_raw_llm = OpenAIAdapter(LLMConfig(
    model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    api_key="e4641e8952ceb9c3c7126675de51f3fb28bbb93b2ce632f13eac11d848f02bd0",
    base_url="https://api.together.xyz/v1",
    temperature=0.2,
    max_tokens=4096,
    timeout=120.0,
))

# Debug wrapper to see raw LLM output
class DebugLLM:
    def __init__(self, inner):
        self.inner = inner
    async def generate(self, messages):
        response = await self.inner.generate(messages)
        print(f"\n  [RAW LLM OUTPUT] ({len(response)} chars):")
        print(f"  ---")
        # Show first 500 chars
        preview = response[:500]
        for line in preview.split("\n"):
            print(f"  | {line}")
        if len(response) > 500:
            print(f"  | ... ({len(response) - 500} more chars)")
        print(f"  ---")
        return response

llm = DebugLLM(_raw_llm)

agent = Agent(
    llm=llm,
    tools=[calculate, lookup_info],
    instructions="You are a helpful assistant that can do math and look up programming topics.",
    max_steps=10,
)

# ---- Event hooks for live observation ----

@agent.on("step_start")
async def on_step_start(data):
    print(f"\n{'='*60}")
    print(f"  STEP {data['step']} starting...")
    print(f"{'='*60}")

@agent.on("step_end")
async def on_step_end(data):
    print(f"  Node: {data['node_id']} | Task: {data['task_id']}")
    print(f"  Tool: {data['tool']}")
    print(f"  Reason: {data['reason']}")
    print(f"  Duration: {data['duration_ms']:.0f}ms")

@agent.on("tool_called")
async def on_tool(data):
    print(f"  >> Tool called: {data['tool']}({data['args']})")

@agent.on("complete")
async def on_done(data):
    print(f"\n{'*'*60}")
    print(f"  COMPLETED in {data['steps']} steps")
    print(f"  Reason: {data['reason']}")
    print(f"{'*'*60}")

@agent.on("error")
async def on_error(data):
    print(f"\n  !! ERROR at step {data['step']}: {data['error']}")


# ---- Run ----

async def main():
    print("="*60)
    print("  RDPGE v0.1.0 — First Live Test")
    print("  LLM: Qwen3 Coder 480B (Together AI)")
    print("="*60)

    task = "Calculate 234 * 567 and also look up what asyncio is."

    print(f"\n  Task: {task}\n")

    result = await agent.run(task)

    print(f"\n{'='*60}")
    print(f"  RESULT")
    print(f"{'='*60}")
    print(f"  Status:     {result.status}")
    print(f"  Success:    {result.success}")
    print(f"  Steps:      {result.steps}")
    print(f"  Session ID: {result.session_id}")
    print(f"  Reason:     {result.reason}")
    if result.error:
        print(f"  Error:      {result.error}")

    # Show trace summary
    summary = result.trace.summary()
    print(f"\n  Trace Summary:")
    print(f"    Total steps:    {summary['total_steps']}")
    print(f"    Tasks created:  {summary['tasks_created']}")
    print(f"    Edges used:     {summary['edges_used']}")
    print(f"    Errors:         {summary['errors']}")

    # Show final graph state
    print(f"\n  Graph State:")
    for node_id, node in result.graph.nodes.items():
        tool_name = node.tool_call.name if node.tool_call else "N/A"
        print(f"    {node_id} (task {node.task_id}): {tool_name}")


if __name__ == "__main__":
    asyncio.run(main())
