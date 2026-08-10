import sys
import subprocess

print("Running Step 8B Common Training Protocol Script...")
res = subprocess.run([sys.executable, "scratch/execute_step8b_protocol.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
