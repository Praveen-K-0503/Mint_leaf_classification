import sys
import subprocess

print("Running Final Verification Gate & Reproducibility Release Script...")
res = subprocess.run([sys.executable, "scratch/execute_final_verification_gate.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
