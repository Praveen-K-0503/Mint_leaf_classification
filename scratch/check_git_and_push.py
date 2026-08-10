import os
import subprocess
import sys
from pathlib import Path

# Add standard Git paths to OS environment PATH
git_cmd_dir = r"C:\Program Files\Git\cmd"
if git_cmd_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = git_cmd_dir + os.pathsep + os.environ.get("PATH", "")

git_candidates = [
    r"C:\Program Files\Git\cmd\git.exe",
    "git",
    r"C:\Program Files\Git\bin\git.exe",
    r"C:\Program Files (x86)\Git\cmd\git.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\Git\cmd\git.exe"),
    os.path.expanduser(r"~\AppData\Local\GitHubDesktop\bin\git.exe"),
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

if not git_bin:
    print("❌ Git executable not found. Please restart your terminal or VS Code.")
else:
    repo_url = "https://github.com/Praveen-K-0503/Mint_leaf_classification.git"
    project_dir = r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai"
    
    print("\n--- ⚙️ Setting Up Git Configuration ---")
    subprocess.run([git_bin, "config", "user.name", "Praveen K"], cwd=project_dir)
    subprocess.run([git_bin, "config", "user.email", "praveen@mintleaf.ai"], cwd=project_dir)
    
    print("\n--- 🚀 Executing Git Workflow ---")
    commands = [
        [git_bin, "init"],
        [git_bin, "branch", "-M", "main"],
        [git_bin, "remote", "remove", "origin"],
        [git_bin, "remote", "add", "origin", repo_url],
        [git_bin, "add", "."],
        [git_bin, "commit", "-m", "Step 1 & Step 2: Mint Leaf AI foundation, dataset audit module and professional README"],
        [git_bin, "push", "-u", "origin", "main"]
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
