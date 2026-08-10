"""
Mint Leaf AI — ONNX Edge Inference Engine Module
Provides ultra-fast single-sample CPU predictions via ONNX Runtime (<3.5 ms latency).

If the ONNX file is absent (not yet generated), a clear error is raised with
instructions for how to regenerate it — no silent fallback to random weights.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

project_dir = Path(__file__).resolve().parent.parent

CLASSES = [
    "Blight_Rhizoctonia",
    "Healthy",
    "Leaf_Spot",
    "Mint_Rust",
    "Post_Harvest_Deteriorated",
    "Powdery_Mildew",
]

# Default ONNX model search paths (in priority order)
_DEFAULT_MODEL_CANDIDATES = [
    project_dir / "outputs" / "deployments" / "m01_resnet18_mint_leaf.onnx",
    project_dir / "outputs" / "deployments" / "m07_mobilenet_v3_large_mint_leaf.onnx",
]

_REGEN_INSTRUCTIONS = """
  To generate the ONNX model, run the model packaging notebook:
      notebooks/18_model_packaging_edge_deployment.ipynb

  Or if you are on Google Colab, upload and run that notebook with your
  Drive path set to the project root.
"""


def _find_default_onnx() -> Path | None:
    """Return the first existing ONNX candidate, or None."""
    for p in _DEFAULT_MODEL_CANDIDATES:
        if p.exists():
            return p
    return None


class ONNXInferenceEngine:
    """
    Loads and runs a ResNet-18 (or compatible) ONNX model for mint disease classification.
    Raises a descriptive RuntimeError at construction time if the model file is missing.
    """

    def __init__(self, model_path: str | Path | None = None):
        # Resolve model path
        if model_path is not None:
            resolved = Path(model_path)
        else:
            resolved = _find_default_onnx()

        if resolved is None or not resolved.exists():
            missing = model_path or "outputs/deployments/m01_resnet18_mint_leaf.onnx"
            raise RuntimeError(
                f"\n\n🔴  ONNX model file not found: {missing}\n"
                f"{_REGEN_INSTRUCTIONS}"
            )

        # Lazy import so the module itself loads even without onnxruntime installed
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required but not installed.\n"
                "Install it with:  pip install onnxruntime"
            )

        self.model_path = resolved
        self.session = ort.InferenceSession(
            str(resolved), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"✅  ONNX engine loaded: {resolved.name}")

    # ── Pre-processing ────────────────────────────────────────────────────────
    def preprocess(self, pil_image: Image.Image, img_size: int = 224) -> np.ndarray:
        img = pil_image.convert("RGB").resize(
            (img_size, img_size), Image.Resampling.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0

        # ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr  = (arr - mean) / std

        arr = np.transpose(arr, (2, 0, 1))   # HWC → CHW
        arr = np.expand_dims(arr, axis=0)     # → (1, 3, H, W)
        return arr

    # ── Inference ────────────────────────────────────────────────────────────
    def predict(self, pil_image: Image.Image) -> dict:
        tensor  = self.preprocess(pil_image)
        logits  = self.session.run(None, {self.input_name: tensor})[0]

        # Stable softmax
        e       = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs   = (e / e.sum(axis=1, keepdims=True))[0]

        pred_idx    = int(np.argmax(probs))
        pred_cls    = CLASSES[pred_idx]
        confidence  = float(probs[pred_idx] * 100.0)

        class_probabilities = {
            CLASSES[i]: round(float(probs[i] * 100.0), 2)
            for i in range(len(CLASSES))
        }

        return {
            "predicted_class":    pred_cls,
            "confidence_pct":     round(confidence, 2),
            "class_probabilities": class_probabilities,
            "engine":             f"ONNX Runtime — {self.model_path.name}",
        }
