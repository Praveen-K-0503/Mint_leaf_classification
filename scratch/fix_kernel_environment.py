import os
import sys
import subprocess

print(f"Current Sys Executable: {sys.executable}")

# Common Windows python executable locations
python_paths = [
    sys.executable,
    r"C:\Program Files\Python311\python.exe",
    r"C:\Users\USER\AppData\Local\Programs\Python\Python311\python.exe",
    r"C:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe",
    "python"
]

packages = ["matplotlib", "seaborn", "opencv-python", "pillow", "pandas", "numpy", "tqdm"]

for py in set(python_paths):
    try:
        print(f"\nChecking & Installing dependencies in: {py}")
        res = subprocess.run([py, "-m", "pip", "install"] + packages + ["--quiet"], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"✅ Successfully installed packages in {py}")
        else:
            print(f"Notice for {py}: {res.stderr.strip()}")
    except Exception as e:
        print(f"Could not run pip for {py}: {e}")
