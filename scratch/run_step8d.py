import sys
import subprocess

print("Running Step 8D Specimen & Provenance Independence Resolution Script...")
res = subprocess.run([sys.executable, "scratch/execute_step8d_specimen_resolution.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
