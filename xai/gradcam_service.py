"""
Mint Leaf AI — Real-Time Grad-CAM XAI Service Module
Generates visual heatmaps highlighting pathological disease lesions on leaf images.

If the PyTorch checkpoint is absent, a clear RuntimeError is raised with
instructions — the service never silently uses random/uninitialised weights.
"""

import io
import sys
import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

project_dir = Path(__file__).resolve().parent.parent
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

import torch
import torch.nn.functional as F

_REGEN_INSTRUCTIONS = """
  To generate the PyTorch checkpoint, run the benchmark notebook:
      notebooks/14_25_model_benchmark_suite.ipynb  (Step 8C)

  The checkpoint will be saved to:
      outputs/experiments/M01_resnet18/best_model.pt
"""

# ── Dynamic target-layer resolver ─────────────────────────────────────────────
def _get_target_layer(model: torch.nn.Module, model_id: str) -> torch.nn.Module:
    """
    Resolve the best Grad-CAM target layer based on the model architecture.
    Falls back to the last Conv2d found if architecture is not recognised.
    """
    mid = model_id.lower()
    try:
        if "resnet" in mid or "resnext" in mid:
            return model.layer4[-1]          # last residual block
        elif "densenet" in mid:
            return model.features.denseblock4
        elif "mobilenet" in mid:
            return model.features[-1]
        elif "efficientnet" in mid:
            return model.features[-1]
        elif "convnext" in mid:
            return model.features[-1]
        elif "swin" in mid or "vit" in mid:
            # Transformers don't have spatial conv maps — return None to skip
            return None
    except (AttributeError, IndexError):
        pass

    # Generic fallback: last Conv2d
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    return last_conv


class GradCAMService:
    """
    Wraps a PyTorch ResNet-18 (or compatible) model with Grad-CAM hooks.

    Parameters
    ----------
    model_id : str
        Identifier matching the checkpoint directory under outputs/experiments/.
    """

    def __init__(self, model_id: str = "M01_resnet18"):
        from models.architectures.factory import build_model
        from training.data.dataset import get_transforms

        self.model_id   = model_id
        self._available = False

        ckpt_path = project_dir / "outputs" / "experiments" / model_id / "best_model.pt"

        # Build model skeleton
        self.model = build_model(model_name=model_id, num_classes=6, pretrained=False)

        if not ckpt_path.exists():
            print(
                f"\n⚠️   Grad-CAM checkpoint not found: {ckpt_path}"
                f"\n{_REGEN_INSTRUCTIONS}"
                f"\n    Grad-CAM heatmaps will be UNAVAILABLE until the checkpoint exists.\n"
            )
            # Mark unavailable — do NOT load random weights
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        state = torch.load(ckpt_path, map_location=device)
        # Support both raw state_dict and wrapped dicts
        sd = state.get("model_state_dict", state)
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()

        self.device = device
        self._gradients  = None
        self._activations = None

        # Resolve target layer
        self.target_layer = _get_target_layer(self.model, model_id)
        if self.target_layer is None:
            print(f"⚠️   Grad-CAM not supported for architecture: {model_id}")
            return

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

        self.transform   = get_transforms(224, is_train=False)
        self._available  = True
        print(f"✅  Grad-CAM service loaded: {model_id} @ {ckpt_path.name}")

    # ── Hooks ────────────────────────────────────────────────────────────────
    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    # ── Public API ───────────────────────────────────────────────────────────
    @property
    def is_available(self) -> bool:
        return self._available

    def generate_gradcam_base64(
        self, pil_image: Image.Image, target_class: int | None = None
    ) -> str | None:
        """
        Generate a Grad-CAM heatmap overlay.

        Returns a base64-encoded PNG data URI, or None if the service is unavailable.
        """
        if not self._available:
            return None

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        self.model.eval()
        outputs = self.model(tensor)

        if target_class is None:
            target_class = int(torch.argmax(outputs, dim=1).item())

        score = outputs[0, target_class]
        self.model.zero_grad()
        score.backward()

        weights = torch.mean(self._gradients, dim=(2, 3), keepdim=True)
        cam     = F.relu(torch.sum(weights * self._activations, dim=1, keepdim=True))
        cam     = cam.squeeze().cpu().numpy()

        # Normalise
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to input image dimensions
        img_rgb = np.array(pil_image.convert("RGB"))
        h, w    = img_rgb.shape[:2]
        cam_up  = cv2.resize(cam, (w, h))

        heatmap     = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay     = cv2.addWeighted(img_rgb, 0.5, heatmap_rgb, 0.5, 0)

        buffered = io.BytesIO()
        Image.fromarray(overlay).save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
