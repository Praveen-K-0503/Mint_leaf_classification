import sys
import subprocess

print("Running Step 7 Framework Validation Script...")
res = subprocess.run([sys.executable, "scratch/execute_step7_framework_validation.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
