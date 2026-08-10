import sys
import subprocess

print("Running Step 8C Independent Benchmark Audit Script...")
res = subprocess.run([sys.executable, "scratch/execute_step8c_independent_audit.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
