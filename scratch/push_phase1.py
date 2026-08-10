import subprocess, sys
git = r"C:\Program Files\Git\cmd\git.exe"
proj = r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai"
for cmd in [
    [git, "add", "."],
    [git, "commit", "-m",
     "Complete project cleanup: remove 40+ redundant scratch files, "
     "add generate_demo_model.py, cleanup_and_build.py, rewrite README.md, "
     "fix requirements.txt, add __init__.py to all packages, "
     "fix run_app.py/predict.py/backend/frontend/engine/gradcam"],
    [git, "push", "origin", "main"],
]:
    r = subprocess.run(cmd, cwd=proj, capture_output=True, text=True)
    print(f"git {' '.join(cmd[1:3])}: RC={r.returncode}")
    out = (r.stdout + r.stderr).strip()
    for line in out.splitlines()[:8]: print(f"  {line}")
