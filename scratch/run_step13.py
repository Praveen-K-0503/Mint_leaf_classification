import sys
import subprocess

print("Running Step 13 Complete Project Synthesis & Final Research Paper Script...")
res = subprocess.run([sys.executable, "scratch/execute_step13_project_synthesis.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
