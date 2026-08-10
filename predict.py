"""
Mint Leaf AI — Standalone CLI Inference Script (v1.0 Release)

Usage:
    python predict.py --image path/to/leaf_image.jpg
    python predict.py --image path/to/leaf_image.jpg --model outputs/deployments/m01_resnet18_mint_leaf.onnx
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# ── Class labels (must match training order) ──────────────────────────────────
CLASSES = [
    "Blight_Rhizoctonia",
    "Healthy",
    "Leaf_Spot",
    "Mint_Rust",
    "Post_Harvest_Deteriorated",
    "Powdery_Mildew",
]

DISPLAY_NAMES = {
    "Blight_Rhizoctonia":       "Blight / Rhizoctonia Rot",
    "Healthy":                  "Healthy Mint (No Disease)",
    "Leaf_Spot":                "Leaf Spot (Septoria)",
    "Mint_Rust":                "Mint Rust (Puccinia menthae)",
    "Post_Harvest_Deteriorated":"Post-Harvest Deterioration",
    "Powdery_Mildew":           "Powdery Mildew (Erysiphe)",
}


def preprocess_image(img_path: Path, img_size: int = 224) -> np.ndarray:
    """Load, resize, and normalise an image to an ONNX-compatible float32 tensor."""
    img = Image.open(img_path).convert("RGB").resize(
        (img_size, img_size), Image.Resampling.BILINEAR
    )
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # ImageNet mean / std normalisation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std

    # HWC → CHW, add batch dimension → (1, 3, 224, 224)
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def find_default_model(project_root: Path) -> Path | None:
    """Return the first existing ONNX model in the standard location, or None."""
    candidates = [
        project_root / "outputs" / "deployments" / "m01_resnet18_mint_leaf.onnx",
        project_root / "outputs" / "deployments" / "m07_mobilenet_v3_large_mint_leaf.onnx",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def predict(image_path: str, model_path: str | None = None) -> None:
    project_root = Path(__file__).resolve().parent

    # ── Resolve model ────────────────────────────────────────────────────────
    if model_path:
        model_p = Path(model_path)
    else:
        model_p = find_default_model(project_root)

    if model_p is None or not model_p.exists():
        print("\n❌  ONNX model file not found.")
        print("    Expected location: outputs/deployments/m01_resnet18_mint_leaf.onnx")
        print("\n    To generate it, re-run notebook:")
        print("    notebooks/18_model_packaging_edge_deployment.ipynb\n")
        sys.exit(1)

    # ── Load ONNX runtime ────────────────────────────────────────────────────
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌  onnxruntime is not installed.")
        print("    Install it with:  pip install onnxruntime")
        sys.exit(1)

    # ── Load image ───────────────────────────────────────────────────────────
    img_p = Path(image_path)
    if not img_p.exists():
        print(f"\n❌  Image not found: {img_p}\n")
        sys.exit(1)

    # ── Run inference ────────────────────────────────────────────────────────
    session    = ort.InferenceSession(str(model_p), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    tensor     = preprocess_image(img_p)
    logits     = session.run(None, {input_name: tensor})[0]
    probs      = softmax(logits)[0]

    pred_idx   = int(np.argmax(probs))
    pred_cls   = CLASSES[pred_idx]
    confidence = float(probs[pred_idx] * 100.0)

    # ── Print result ─────────────────────────────────────────────────────────
    print()
    print("=======================================================")
    print("🌿  MINT LEAF AI — DIAGNOSTIC RESULT  (v1.0)")
    print("=======================================================")
    print(f"  Image      : {img_p.name}")
    print(f"  Model      : {model_p.name}")
    print(f"  Diagnosis  : {DISPLAY_NAMES[pred_cls]}")
    print(f"  Confidence : {confidence:.2f}%")
    print("-------------------------------------------------------")
    print("  Class Probabilities:")
    for i, cls in enumerate(CLASSES):
        bar = "█" * int(probs[i] * 40)
        print(f"  {DISPLAY_NAMES[cls]:<30} {probs[i]*100:6.2f}%  {bar}")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mint Leaf AI — Standalone ONNX Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python predict.py --image data/processed/test/Healthy/img.jpg",
    )
    parser.add_argument("--image", required=True,
                        help="Path to the input mint leaf image.")
    parser.add_argument("--model", default=None,
                        help="Path to the ONNX model file (auto-detected if omitted).")
    args = parser.parse_args()
    predict(args.image, args.model)
