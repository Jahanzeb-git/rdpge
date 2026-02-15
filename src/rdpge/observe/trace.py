# trace.py — Execution Trace Collector
#
# Records every step of the agent's execution for:
#   - Debugging
#   - Performance analysis
#   - Cost tracking
#   - Observability dashboards

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class StepTrace:
    """One step in the agent's execution trace."""
    step_number: int
    node_id: str
    task_id: str
    reason: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_output_length: int = 0
    console_output_length: int = 0
    code_length: int = 0
    edge_restored: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: float = 0.0


@dataclass
class SessionTrace:
    """Complete trace of an agent session."""
    session_id: str
    original_request: str
    steps: list[StepTrace] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    total_steps: int = 0
    tasks_created: int = 0
    edges_used: int = 0
    errors: int = 0

    def add_step(self, step: StepTrace) -> None:
        """Add a step trace to the session."""
        self.steps.append(step)
        self.total_steps = len(self.steps)
        if step.edge_restored:
            self.edges_used += 1
        if not step.success:
            self.errors += 1

    def finalize(self) -> None:
        """Mark the session as complete."""
        self.end_time = datetime.now().isoformat()
        # Count unique tasks
        unique_tasks = set(s.task_id for s in self.steps)
        self.tasks_created = len(unique_tasks)

    def summary(self) -> dict:
        """Return a summary of the session for logging/display."""
        return {
            "session_id": self.session_id,
            "request": self.original_request[:100],
            "total_steps": self.total_steps,
            "tasks_created": self.tasks_created,
            "edges_used": self.edges_used,
            "errors": self.errors,
            "start": self.start_time,
            "end": self.end_time,
        }
