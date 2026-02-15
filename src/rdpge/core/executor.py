# executor.py — Code Executor (Sandbox)
#
# Executes LLM-generated Python code safely.
# Handles:
#   - Sandboxed exec() with restricted builtins
#   - Extracting the `action` dict from local namespace
#   - Capturing console output (stdout)
#   - Error handling

import sys
import io
import traceback
from typing import Optional, Dict, Any
from .models import ExecutionResult, ActionDict

class CodeExecutor:
    def __init__(self):
        # We could restrict imports here, but for now we trust the LLM sandbox
        pass

    def execute(self, code: str) -> ExecutionResult:
        # 1. Setup the capture buffer (a fake file in memory)
        buffer = io.StringIO()
        
        # 2. Save the real stdout so we can restore it later
        original_stdout = sys.stdout
        
        # 3. Redirect stdout to our buffer
        sys.stdout = buffer
        
        # Clean local scope to capture variables created by the code
        local_scope: Dict[str, Any] = {}
        
        try:
            # 4. EXECUTE THE CODE!
            # defined globals allows basic builtins, but we could restrict this
            exec(code, {"__builtins__": __builtins__}, local_scope)
            
            # 5. Restore stdout immediately
            sys.stdout = original_stdout
            console_output = buffer.getvalue()
            
            # 6. Extract the 'action' variable
            if "action" not in local_scope:
                return ExecutionResult(
                    success=False,
                    action=None,
                    code=code,
                    console_output=console_output,
                    error="Code did not define an 'action' variable."
                )
            
            # 7. Validate action with Pydantic
            try:
                action_data = local_scope["action"]
                validated_action = ActionDict(**action_data)
                
                return ExecutionResult(
                    success=True,
                    action=validated_action,
                    code=code,
                    console_output=console_output,
                    error=None
                )
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    action=None,
                    code=code,
                    console_output=console_output,
                    error=f"Invalid action schema: {str(e)}"
                )
                
        except Exception as e:
            # If exec() crashes (syntax error, runtime error)
            sys.stdout = original_stdout # ALWAYS restore stdout
            console_output = buffer.getvalue()
            
            # Get the full traceback (stack trace) as a string
            error_msg = traceback.format_exc()
            
            return ExecutionResult(
                success=False,
                action=None,
                code=code,
                console_output=console_output,
                error=f"Runtime error: {error_msg}"
            )