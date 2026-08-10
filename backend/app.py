"""
Mint Leaf AI — FastAPI REST Web Server  (v1.0)

Endpoints:
  GET  /              → serves the interactive frontend dashboard
  GET  /api/health    → system health + model status
  POST /api/diagnose  → full pipeline: quality → ONNX → Grad-CAM → advisory
  GET  /api/benchmark → 25-model benchmark leaderboard (CSV-backed)

Startup behaviour:
  - If the ONNX model is missing the server STILL starts, but /api/diagnose
    returns a 503 with clear instructions on how to regenerate the model.
  - If the Grad-CAM checkpoint is missing the endpoint still returns a
    diagnosis + advisory; the heatmap field is null and a notice is added.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

from PIL import Image

# ── Project root on sys.path ──────────────────────────────────────────────────
project_dir = Path(__file__).resolve().parent.parent
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

# ── FastAPI imports ────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ── Internal modules ──────────────────────────────────────────────────────────
from utils.image_checker import check_image_quality
from recommendation.advisory import get_agronomic_advisory

# ── Lazy engine initialisation ────────────────────────────────────────────────
# Engines are initialised once at startup.  If files are missing the server
# stays up; /api/health reveals the degraded state; /api/diagnose returns 503.

_inference_engine = None
_xai_service      = None
_startup_errors   = []


def _init_inference_engine():
    global _inference_engine, _startup_errors
    try:
        from inference.engine import ONNXInferenceEngine
        _inference_engine = ONNXInferenceEngine()
    except (RuntimeError, ImportError, FileNotFoundError) as exc:
        msg = str(exc)
        _startup_errors.append(f"ONNX engine: {msg}")
        print(f"\n⚠️  Inference engine unavailable — {msg}\n")


def _init_xai_service():
    global _xai_service, _startup_errors
    try:
        from xai.gradcam_service import GradCAMService
        _xai_service = GradCAMService(model_id="M01_resnet18")
    except Exception as exc:
        msg = str(exc)
        _startup_errors.append(f"Grad-CAM service: {msg}")
        print(f"\n⚠️  Grad-CAM service unavailable — {msg}\n")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mint Leaf AI — Edge Diagnostic Server",
    description=(
        "Real-Time Explainable AI Diagnostic & Treatment Advisory API "
        "for Mint Plant Pathology (ResNet-18, ONNX Runtime, Grad-CAM)"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
frontend_dir = project_dir / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.on_event("startup")
async def startup_event():
    print("\n🌿  Mint Leaf AI — starting up...")
    _init_inference_engine()
    _init_xai_service()
    status = "✅ FULLY OPERATIONAL" if not _startup_errors else "⚠️  DEGRADED (see /api/health)"
    print(f"    Server status: {status}\n")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_index():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return (
        "<h2>🌿 Mint Leaf AI API is running</h2>"
        "<p>Frontend: place <code>index.html</code> in <code>frontend/</code></p>"
        "<p><a href='/docs'>→ API Documentation</a></p>"
    )


@app.get("/api/health")
async def health_check():
    return {
        "status":           "DEGRADED" if _startup_errors else "ONLINE",
        "system":           "Mint Leaf AI v1.0",
        "inference_engine": "ONLINE" if _inference_engine else "OFFLINE — ONNX model missing",
        "xai_service":      (
            "ONLINE" if (_xai_service and _xai_service.is_available)
            else "OFFLINE — checkpoint missing"
        ),
        "startup_errors":   _startup_errors,
        "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "regen_tip": (
            None if not _startup_errors else
            "Run notebooks/14_25_model_benchmark_suite.ipynb (Step 8C) to get .pt, "
            "then notebooks/18_model_packaging_edge_deployment.ipynb (Step 12) to get .onnx"
        ),
    }


@app.post("/api/diagnose")
async def diagnose_leaf(file: UploadFile = File(...)):
    # Gate: refuse if ONNX engine is not loaded
    if _inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "ONNX inference engine is offline. "
                "Run notebook 18_model_packaging_edge_deployment.ipynb to generate "
                "outputs/deployments/m01_resnet18_mint_leaf.onnx, then restart the server."
            ),
        )

    # Validate content type
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file (jpg, png, webp).")

    # Read + decode
    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:   # 15 MB hard limit
        raise HTTPException(status_code=413, detail="Image file too large (max 15 MB).")

    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    t0 = time.perf_counter()

    # ── 1. Image quality pre-screening ───────────────────────────────────────
    quality_res = check_image_quality(pil_image)

    # ── 2. ONNX inference ────────────────────────────────────────────────────
    pred_res = _inference_engine.predict(pil_image)

    # ── 3. Grad-CAM heatmap (best-effort) ───────────────────────────────────
    xai_base64  = None
    xai_notice  = None
    if _xai_service and _xai_service.is_available:
        xai_base64 = _xai_service.generate_gradcam_base64(pil_image)
    else:
        xai_notice = (
            "Grad-CAM unavailable: PyTorch checkpoint not found. "
            "Run notebook 14_25_model_benchmark_suite.ipynb (Step 8C) to generate it."
        )

    # ── 4. Agronomic advisory ────────────────────────────────────────────────
    advisory_res = get_agronomic_advisory(pred_res["predicted_class"])

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "diagnostic_result": {
            "predicted_class":    pred_res["predicted_class"],
            "disease_display_name": advisory_res["disease_name"],
            "confidence_pct":     pred_res["confidence_pct"],
            "class_probabilities": pred_res["class_probabilities"],
        },
        "quality_prescreening": quality_res,
        "xai_heatmap_base64":   xai_base64,
        "xai_notice":           xai_notice,
        "agronomic_advisory":   advisory_res,
        "performance_metrics": {
            "total_request_latency_ms": elapsed_ms,
            "engine": pred_res["engine"],
        },
    }


@app.get("/api/benchmark")
async def get_benchmark_matrix():
    master_csv = project_dir / "outputs" / "reports" / "model_suite" / "master_model_comparison.csv"
    if not master_csv.exists():
        return {
            "data":   [],
            "notice": (
                "Benchmark CSV not yet generated. "
                "Run notebook 14_25_model_benchmark_suite.ipynb (Step 8C) to produce it."
            ),
        }
    import pandas as pd
    df = pd.read_csv(master_csv)
    return {"data": df.to_dict(orient="records"), "notice": None}
