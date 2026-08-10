import sys
import subprocess

print("Running Step 12 Model Packaging & Edge Deployment Integration Script...")
res = subprocess.run([sys.executable, "scratch/execute_step12_edge_deployment.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
