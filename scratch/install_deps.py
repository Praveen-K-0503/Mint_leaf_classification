import os
import sys
import subprocess

print(f"Executing Python Environment: {sys.executable}")
print("Installing required dependencies (matplotlib, seaborn, opencv-python, pillow, pandas, numpy, tqdm)...")

packages = [
    "matplotlib",
    "seaborn",
    "opencv-python",
    "pillow",
    "pandas",
    "numpy",
    "tqdm"
]

try:
    res = subprocess.run([sys.executable, "-m", "pip", "install"] + packages, capture_output=True, text=True)
    print("Pip Install STDOUT:", res.stdout)
    if res.stderr:
        print("Pip Install STDERR:", res.stderr)
except Exception as e:
    print("Failed to install packages via pip:", e)
