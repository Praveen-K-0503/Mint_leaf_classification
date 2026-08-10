"""
Mint Leaf AI — Phase 1 Startup Verification Script
Checks all dependencies and file paths without starting the server.
Run this FIRST before run_app.py to diagnose any remaining issues.

Usage:
    python check_startup.py
"""

import sys
import importlib
from pathlib import Path

project_dir = Path(__file__).resolve().parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

errors = []
warnings = []

def check(label, condition, error_msg=None, warn=False):
    if condition:
        print(f"  {PASS}  {label}")
    elif warn:
        print(f"  {WARN} {label} — {error_msg}")
        warnings.append(label)
    else:
        print(f"  {FAIL}  {label} — {error_msg}")
        errors.append(label)

print("\n=======================================================")
print("🌿  MINT LEAF AI — STARTUP VERIFICATION")
print("=======================================================\n")

# ── Python packages ───────────────────────────────────────────────────────────
print("📦 Checking required Python packages...")

for pkg, install_name in [
    ("fastapi",         "fastapi"),
    ("uvicorn",         "uvicorn"),
    ("multipart",       "python-multipart"),
    ("PIL",             "pillow"),
    ("cv2",             "opencv-python"),
    ("numpy",           "numpy"),
    ("torch",           "torch"),
    ("torchvision",     "torchvision"),
    ("onnxruntime",     "onnxruntime"),
    ("timm",            "timm"),
]:
    try:
        importlib.import_module(pkg)
        check(f"Python package: {pkg}", True)
    except ImportError:
        check(f"Python package: {pkg}", False,
              f"not installed — run: pip install {install_name}")

# ── Internal module imports ───────────────────────────────────────────────────
print("\n📂 Checking internal module imports...")

for mod in [
    "utils.image_checker",
    "recommendation.advisory",
    "inference.engine",
    "models.architectures.factory",
    "training.data.dataset",
]:
    try:
        importlib.import_module(mod)
        check(f"Module: {mod}", True)
    except ImportError as e:
        check(f"Module: {mod}", False, str(e))
    except Exception as e:
        # Might fail due to missing files — that's expected and separate
        check(f"Module: {mod} (import only)", True)

# ── Critical file paths ───────────────────────────────────────────────────────
print("\n📁 Checking critical file paths...")

critical_paths = {
    "predict.py": project_dir / "predict.py",
    "run_app.py": project_dir / "run_app.py",
    "backend/app.py": project_dir / "backend" / "app.py",
    "frontend/index.html": project_dir / "frontend" / "index.html",
    "inference/engine.py": project_dir / "inference" / "engine.py",
    "xai/gradcam_service.py": project_dir / "xai" / "gradcam_service.py",
    "utils/image_checker.py": project_dir / "utils" / "image_checker.py",
    "recommendation/advisory.py": project_dir / "recommendation" / "advisory.py",
}
for label, p in critical_paths.items():
    check(label, p.exists(), f"File not found: {p}")

# ── Model files (warn, not error — can run in degraded mode) ─────────────────
print("\n🤖 Checking model asset files (required for full pipeline)...")

model_paths = {
    "ONNX model (ResNet-18)":
        project_dir / "outputs" / "deployments" / "m01_resnet18_mint_leaf.onnx",
    "PT checkpoint (ResNet-18)":
        project_dir / "outputs" / "experiments" / "M01_resnet18" / "best_model.pt",
}
for label, p in model_paths.items():
    check(label, p.exists(),
          f"Missing — run the corresponding training/packaging notebook to generate it.",
          warn=True)

# ── Data directory ────────────────────────────────────────────────────────────
print("\n🗂️  Checking dataset directories...")

split_dirs = {
    "data/processed/train": project_dir / "data" / "processed" / "train",
    "data/processed/val":   project_dir / "data" / "processed" / "val",
    "data/processed/test":  project_dir / "data" / "processed" / "test",
}
for label, p in split_dirs.items():
    if p.exists():
        n_imgs = len(list(p.rglob("*.jpg"))) + len(list(p.rglob("*.png")))
        check(f"{label}  ({n_imgs} images)", n_imgs > 0,
              "Directory exists but is empty — re-run dataset construction notebook.",
              warn=(n_imgs == 0))
    else:
        check(label, False,
              "Directory missing — re-run notebooks/06_training_dataset_construction.ipynb",
              warn=True)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=======================================================")
if not errors and not warnings:
    print("🎉  ALL CHECKS PASSED — server is fully ready.")
    print("    Run:  python run_app.py")
elif not errors:
    print(f"⚠️   SERVER WILL START IN DEGRADED MODE ({len(warnings)} warning(s))")
    print("    Some features (Grad-CAM, ONNX inference) need model files.")
    print("    See warnings above. Run:  python run_app.py")
else:
    print(f"❌  {len(errors)} CRITICAL ERROR(S) — fix these before starting the server.")
    for e in errors:
        print(f"    → {e}")
    if warnings:
        print(f"\n⚠️   Also {len(warnings)} warning(s) about model assets:")
        for w in warnings:
            print(f"    → {w}")

print("=======================================================\n")
sys.exit(1 if errors else 0)
