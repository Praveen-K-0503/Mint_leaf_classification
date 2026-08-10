import sys
import subprocess

print("Running Step 8C Final Leakage Gate Verification Script...")
res = subprocess.run([sys.executable, "scratch/execute_step8c_final_gate.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
