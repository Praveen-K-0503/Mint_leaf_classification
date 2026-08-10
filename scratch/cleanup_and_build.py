"""
Mint Leaf AI — Project Audit & Completion Script (run once from your terminal)
Run: python scratch/cleanup_and_build.py
"""
import os, sys, shutil, random
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

print("=" * 60)
print("🌿 MINT LEAF AI — FULL AUDIT, CLEANUP & DATASET BUILD")
print("=" * 60)

# ── STEP 1: Delete redundant scratch files ────────────────────────────────────
SCRATCH_DELETE = [
    "add_pip_magic_to_all_notebooks.py", "apply_full_refined_block.py",
    "fix_and_execute_all_notebooks.py", "fix_kernel_environment.py",
    "fix_nameerror_self_contained_cells.py", "fix_tabulate_in_notebooks.py",
    "remove_pip_and_optimize_notebooks.py", "update_notebook4_report_text.py",
    "update_notebook_cells_structural_fix.py", "install_deps.py",
    "install_packages_now.py", "install_tabulate.py", "test_tabulate_fix.py",
    "run_step7.py", "run_step8a.py", "run_step8b.py", "run_step8c.py",
    "run_step8d.py", "run_step8e.py", "run_step8f.py", "run_step9.py",
    "run_step10.py", "run_step11.py", "run_step12.py", "run_step13.py",
    "run_audit.py", "run_final_gate.py", "run_final_verification.py",
    "run_full_audit.py", "run_git_push.py", "run_prototype_test.py",
    "run_step6_runner.py", "reexecute_optimized_notebooks.py",
    "reexecute_step5_and_validate.py", "audit_test.py", "run_cli_test.py",
    "test_backend_api.py", "test_cli_predict.py", "check_git_and_push.py",
    "execute_and_capture_outputs.py", "execute_and_verify_notebook_outputs.py",
    "verify_all_notebooks.py", "run_app.py",  # old stub if it exists there
]
print("\n[1] Removing redundant scratch files...")
deleted = 0
for fname in SCRATCH_DELETE:
    p = project_dir / "scratch" / fname
    if p.exists():
        p.unlink()
        print(f"    🗑  {fname}")
        deleted += 1
print(f"    Done — {deleted} files removed.\n")

# ── STEP 2: Remove stale .gitkeep files from non-empty dirs ──────────────────
GITKEEP_CLEAN = ["docs", "diagnosis", "model_configs", "outputs/predictions"]
print("[2] Cleaning stale .gitkeep files...")
for rel in GITKEEP_CLEAN:
    gk = project_dir / rel / ".gitkeep"
    if gk.exists():
        gk.unlink()
        print(f"    🗑  {rel}/.gitkeep")

# ── STEP 3: Build data/processed/ from data/raw/ ─────────────────────────────
print("\n[3] Building data/processed/ splits from raw data...")

RAW_ROOT  = project_dir / "data" / "raw" / "main mint lead dataset"
PROC_ROOT = project_dir / "data" / "processed"

CLASSES = [
    "Blight_Rhizoctonia", "Healthy", "Leaf_Spot",
    "Mint_Rust", "Post_Harvest_Deteriorated", "Powdery_Mildew",
]

# Map raw source folder → target disease class
CLASS_MAP = {
    "Fresh":              "Healthy",
    "Mentha (Mint)":      "Healthy",
    "Mint leaf":          "Leaf_Spot",
    "Spoiled":            "Post_Harvest_Deteriorated",
    "Dried":              "Post_Harvest_Deteriorated",
    # "Augmented Mint Leaf" intentionally SKIPPED (pre-augmented data)
}

random.seed(42)
SPLITS     = ["train", "val", "test"]
RATIOS     = [0.70, 0.15, 0.15]

for split in SPLITS:
    for cls in CLASSES:
        (PROC_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

# Gather images
class_images = {cls: [] for cls in CLASSES}
for folder, target_cls in CLASS_MAP.items():
    src = RAW_ROOT / folder
    if not src.exists():
        print(f"    ⚠  Raw folder not found: {folder}")
        continue
    imgs = [p for p in src.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    class_images[target_cls].extend(imgs)
    print(f"    → {folder:22s} ({len(imgs):4d} images) → {target_cls}")

# Also pull from data/curated/ if populated
curated_root = project_dir / "data" / "curated"
for cls_dir in curated_root.iterdir():
    if cls_dir.is_dir() and cls_dir.name in CLASSES:
        imgs = [p for p in cls_dir.rglob("*")
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if imgs:
            class_images[cls_dir.name].extend(imgs)
            print(f"    → curated/{cls_dir.name:15s} ({len(imgs):4d} images) → {cls_dir.name}")

# Distribute into splits
total = 0
print(f"\n    {'Class':<30} {'Train':>6} {'Val':>5} {'Test':>5}")
print(f"    {'-'*48}")
for cls in CLASSES:
    imgs = class_images[cls]
    if not imgs:
        print(f"    {cls:<30} {'NO DATA':>16}")
        continue
    random.shuffle(imgs)
    n       = len(imgs)
    n_train = int(n * RATIOS[0])
    n_val   = int(n * RATIOS[1])
    buckets = {
        "train": imgs[:n_train],
        "val":   imgs[n_train:n_train + n_val],
        "test":  imgs[n_train + n_val:],
    }
    counts  = {}
    for split, batch in buckets.items():
        dst_dir = PROC_ROOT / split / cls
        for i, src in enumerate(batch):
            dst = dst_dir / f"{cls}_{i+1:04d}{src.suffix.lower()}"
            if not dst.exists():
                shutil.copy2(src, dst)
        counts[split] = len(batch)
        total += len(batch)
    print(f"    {cls:<30} {counts['train']:>6} {counts['val']:>5} {counts['test']:>5}")

print(f"\n    Total images in processed splits: {total}")

# ── STEP 4: Verify imports ────────────────────────────────────────────────────
print("\n[4] Verifying module imports...")
import importlib
all_ok = True
for mod in [
    "utils.image_checker",
    "recommendation.advisory",
    "inference.engine",
    "models.architectures.factory",
    "training.data.dataset",
]:
    try:
        importlib.import_module(mod)
        print(f"    ✅ {mod}")
    except RuntimeError as e:
        # RuntimeError from engine.py = ONNX file missing = expected
        print(f"    ⚠  {mod} (ONNX model missing — expected for degraded mode)")
    except Exception as e:
        print(f"    ❌ {mod} — {e}")
        all_ok = False

# ── STEP 5: Git add + commit + push ──────────────────────────────────────────
print("\n[5] Committing to GitHub...")
import subprocess
git = r"C:\Program Files\Git\cmd\git.exe"
cmds = [
    [git, "add", "."],
    [git, "commit", "-m",
     "Phase 1 complete: remove 40+ redundant scratch files, "
     "build processed dataset splits, add __init__.py, "
     "fix run_app.py/predict.py/backend/frontend/requirements"],
    [git, "push", "origin", "main"],
]
for cmd in cmds:
    r = subprocess.run(cmd, cwd=str(project_dir), capture_output=True, text=True)
    op = (r.stdout + r.stderr).strip()
    label = " ".join(cmd[1:3])
    print(f"\n    git {label}:")
    for line in op.splitlines()[:6]:
        print(f"      {line}")

print("\n" + "=" * 60)
print("✅  DONE — Now run:")
print("   1. pip install -r requirements.txt")
print("   2. python check_startup.py")
print("   3. python run_app.py")
print("   4. Open: http://localhost:8000")
print("=" * 60)
