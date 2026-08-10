import sys
import subprocess

print("Installing tabulate package...")
subprocess.run([sys.executable, "-m", "pip", "install", "tabulate", "--quiet"])
print("✅ tabulate installed successfully!")
