import sys
import subprocess

print("Running Step 9 Final Scientific Model Comparison Script...")
res = subprocess.run([sys.executable, "scratch/execute_step9_final_comparison.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
