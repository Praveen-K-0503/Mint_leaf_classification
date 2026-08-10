import sys
import subprocess

print("Running Step 11 Error & Robustness Stress-Testing Script...")
res = subprocess.run([sys.executable, "scratch/execute_step11_error_robustness.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
