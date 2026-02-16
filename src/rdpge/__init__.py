# RDPGE — Runtime Dynamic & Probabilistic Graph Execution
#
# A novel agentic AI framework that uses:
#   - Code-as-interface (LLM outputs Python, not JSON tool calls)
#   - Probabilistic graph (dynamic node/task creation)
#   - Context blurring (task-level context management)
#   - Edge restoration (selective memory recall)
#
# Quick Start:
#   from rdpge import Agent, tool
#   from rdpge.llm import OpenAIAdapter, LLMConfig
#
#   @tool()
#   def read_file(path: str) -> str:
#       """Read file contents from disk."""
#       return open(path).read()
#
#   agent = Agent(
#       llm=OpenAIAdapter(LLMConfig(model="gpt-4o", api_key="...")),
#       tools=[read_file],
#       instructions="You are a code assistant.",
#   )
#   result = await agent.run("Fix the bug in auth.py")

from .agent import Agent
from .tools.base import tool, ToolWrapper, BaseTool, ToolSpec
from .core.engine import AgentResult
from .core.models import Message, GraphState, NodeState, SIGNAL_TOOLS
from .storage.base import SessionStore, InMemoryStore

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "tool",
    "ToolWrapper",
    "BaseTool",
    "ToolSpec",
    "AgentResult",
    "Message",
    "GraphState",
    "NodeState",
    "SIGNAL_TOOLS",
    "SessionStore",
    "InMemoryStore",
]

