import sys
import subprocess

print("Running Step 8A 25-Model Registry Verification Script...")
res = subprocess.run([sys.executable, "scratch/execute_step8a_registry.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
