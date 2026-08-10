import sys
import subprocess

print(f"Current Executable: {sys.executable}")
print("Executing pip install directly...")

cmd = [sys.executable, "-m", "pip", "install", "matplotlib", "seaborn", "opencv-python", "pillow", "pandas", "numpy", "tqdm"]
try:
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
except Exception as e:
    print("Error:", e)
