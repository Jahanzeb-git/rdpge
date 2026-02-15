import sys
import io

print("--- STARTING VISUALIZATION ---")

# 1. The Code from LLM
llm_code = """
print("I am running inside the sandbox!")
x = 10 + 5
action = {
    "node": "node-a1",
    "reason": "I calculated x",
    "result": x
}
print(f"Calculation done. x is {x}")
"""
print(f"1. RAW CODE STRING:\n{llm_code}\n--------------------------------")

# 2. Setup Buffer
buffer = io.StringIO()
print(f"2. Created Buffer: {buffer} (It's empty right now)")

# 3. Save Original Stdout
original_stdout = sys.stdout
print(f"3. Saved Original Stdout: {original_stdout} (This is your terminal)")

# 4. Redirect Stdout
print("4. Redirecting stdout to buffer... (You won't see prints after this!)")
sys.stdout = buffer

# --- FROM HERE, PRINT() GOES TO MEMORY, NOT SCREEN ---
print("You cannot see this message in the terminal.")
print("It is going into the StringIO buffer.")

# 5. Execute Code
local_scope = {}
try:
    exec(llm_code, {"__builtins__": __builtins__}, local_scope)
except Exception as e:
    print(f"Error: {e}")

# --- RESTORE STDOUT ---
sys.stdout = original_stdout
print("5. Stdout restored! You can see me again.")

# 6. Check Buffer
print(f"6. Buffer Contents (What was captured):\n---\n{buffer.getvalue()}\n---")

# 7. Check Local Scope
print("7. Extracted Variables (local_scope):")
for key, value in local_scope.items():
    print(f"   - {key}: {value}")

print("\n--- VISUALIZATION COMPLETE ---")
