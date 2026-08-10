import os
import subprocess
import sys
from pathlib import Path

# Add standard Git path to PATH
git_cmd_dir = r"C:\Program Files\Git\cmd"
if git_cmd_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = git_cmd_dir + os.pathsep + os.environ.get("PATH", "")

git_candidates = [
    r"C:\Program Files\Git\cmd\git.exe",
    "git"
]

git_bin = None
for candidate in git_candidates:
    try:
        res = subprocess.run([candidate, "--version"], capture_output=True, text=True)
        if res.returncode == 0:
            git_bin = candidate
            print(f"✅ Found Git executable: {git_bin} ({res.stdout.strip()})")
            break
    except Exception:
        continue

if git_bin:
    project_dir = r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai"
    
    print("\n--- 🚀 Executing Git Commit & Push for Step 5 ---")
    commands = [
        [git_bin, "add", "."],
        [git_bin, "commit", "-m", "Step 5: Controlled Dataset Acquisition and Curation Module"],
        [git_bin, "push", "origin", "main"]
    ]
    
    for cmd in commands:
        try:
            print(f"Executing: {' '.join(cmd)}")
            res = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True)
            if res.stdout and res.stdout.strip():
                print("  [Output]:", res.stdout.strip())
            if res.stderr and res.stderr.strip() and "Switched to a new branch" not in res.stderr and "Reinitialized" not in res.stderr and "To https" not in res.stderr:
                print("  [Notice]:", res.stderr.strip())
        except Exception as e:
            print(f"Execution error: {e}")
