import sys
import subprocess

print("Running Step 8F External Source Domain Generalization Validation Script...")
res = subprocess.run([sys.executable, "scratch/execute_step8f_domain_validation.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
