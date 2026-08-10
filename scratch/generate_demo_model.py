"""
Mint Leaf AI — Demo Model Generator
Creates a lightweight demo ResNet-18 in ONNX format from scratch
(random weights, just enough to prove the full pipeline runs end-to-end).

Run this to get the app working TODAY without waiting hours for training.
Replace with real trained weights later.

Usage: python scratch/generate_demo_model.py
"""
import sys, io, struct
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

print("=" * 55)
print("🌿 MINT LEAF AI — DEMO ONNX MODEL GENERATOR")
print("=" * 55)

# ── Check dependencies ─────────────────────────────────────────────────────
missing = []
for pkg in ["torch", "torchvision", "onnx", "onnxruntime"]:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"\n❌  Missing packages: {', '.join(missing)}")
    print(f"    Install with: pip install {' '.join(missing)}")
    sys.exit(1)

import torch
import torchvision.models as tvm

NUM_CLASSES = 6
IMG_SIZE    = 224

# ── Build ResNet-18 with 6-class head ──────────────────────────────────────
print("\n[1] Building ResNet-18 (random weights, 6 classes)...")
model = tvm.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.eval()

dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
with torch.no_grad():
    out = model(dummy)
    assert out.shape == (1, NUM_CLASSES), f"Bad output shape: {out.shape}"
print(f"    ✅ Model output shape: {out.shape}")

# ── Export ONNX ─────────────────────────────────────────────────────────────
deploy_dir = project_dir / "outputs" / "deployments"
deploy_dir.mkdir(parents=True, exist_ok=True)

onnx_path  = deploy_dir / "m01_resnet18_mint_leaf.onnx"

print(f"\n[2] Exporting ONNX model → {onnx_path.name} ...")
torch.onnx.export(
    model, dummy, str(onnx_path),
    opset_version=14,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    do_constant_folding=True,
)
print(f"    ✅ ONNX exported: {onnx_path.stat().st_size / 1e6:.1f} MB")

# ── Verify ONNX ──────────────────────────────────────────────────────────────
import onnxruntime as ort
import numpy as np

print(f"\n[3] Verifying ONNX Runtime inference...")
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
dummy_np = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
logits = sess.run(None, {inp_name: dummy_np})[0]
assert logits.shape == (1, NUM_CLASSES)
print(f"    ✅ ONNX Runtime OK — output shape: {logits.shape}")

# ── Also save a PyTorch checkpoint for Grad-CAM ──────────────────────────────
exp_dir = project_dir / "outputs" / "experiments" / "M01_resnet18"
exp_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = exp_dir / "best_model.pt"
torch.save({"model_state_dict": model.state_dict(), "demo": True}, ckpt_path)
print(f"\n[4] Saved PyTorch checkpoint → {ckpt_path.relative_to(project_dir)}")

print(f"""
⚠️  NOTE: This model has RANDOM weights — it is a DEMO only.
    Predictions will not be medically meaningful.
    To get real predictions:
      → Run: notebooks/09_train_25_models.ipynb  (needs GPU, ~1 hour)
      → Then: notebooks/18_model_packaging_edge_deployment.ipynb

✅  But the FULL WEB APP PIPELINE will now work end-to-end.
    Run: python run_app.py → http://localhost:8000
""")
print("=" * 55)
