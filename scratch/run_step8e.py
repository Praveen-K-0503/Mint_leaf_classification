import sys
import subprocess

print("Running Step 8E Robustness & Statistical Validation Script...")
res = subprocess.run([sys.executable, "scratch/execute_step8e_statistical_validation.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
