import sys
import subprocess

print("Running Step 8C Full 25-Model Controlled Training Benchmark Script...")
res = subprocess.run([sys.executable, "scratch/execute_step8c_25models.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
