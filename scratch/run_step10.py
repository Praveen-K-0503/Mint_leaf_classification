import sys
import subprocess

print("Running Step 10 XAI Grad-CAM Interpretability Engine Script...")
res = subprocess.run([sys.executable, "scratch/execute_step10_xai.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
