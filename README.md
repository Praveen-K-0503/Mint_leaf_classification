# 🌿 Mint Leaf AI — Real-Time Edge AI Diagnostic System

**Automated detection and advisory for 6 mint leaf pathology classes using a validated deep-learning pipeline with ONNX edge deployment, Grad-CAM XAI, and agronomic treatment recommendations.**

---

## 📌 Project Summary

| Attribute | Value |
|---|---|
| **Primary Model** | ResNet-18 (ONNX Runtime, CPU) |
| **Target Classes** | Healthy, Mint Rust, Powdery Mildew, Leaf Spot, Blight/Rhizoctonia, Post-Harvest Deteriorated |
| **Test Accuracy** | 99.68% (313-image specimen-aware test set) |
| **External F1** | 0.9842 (349 out-of-domain images) |
| **Inference Latency** | 3.45 ms / sample (CPU, ONNX) |
| **Calibration ECE** | 0.32% |

---

## 🚀 Quick Start (Run the App)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Check all systems are ready
```bash
python check_startup.py
```

### Step 3 — Generate model files (choose one)

**Option A — Demo mode (instant, random weights, for UI testing only):**
```bash
python scratch/generate_demo_model.py
```

**Option B — Real trained model (requires GPU, ~1 hour):**
```bash
# Open and run ALL cells in these notebooks in order:
notebooks/09_train_25_models.ipynb
notebooks/18_model_packaging_edge_deployment.ipynb
```

### Step 4 — Clean up and build dataset
```bash
python scratch/cleanup_and_build.py
```

### Step 5 — Launch the web app
```bash
python run_app.py
```

**Open in browser:** http://localhost:8000

---

## 🌐 Web Application Features

| Feature | Description |
|---|---|
| **Drag-and-Drop Upload** | Upload any mint leaf photo |
| **Live Camera Capture** | Use webcam directly in browser |
| **AI Diagnosis** | Real-time classification with confidence scores |
| **Image Quality Gate** | Sharpness + exposure pre-screening |
| **Grad-CAM Heatmap** | Toggle XAI overlay highlighting disease regions |
| **Agronomic Advisory** | Organic, chemical, and cultural treatment cards |
| **25-Model Leaderboard** | Benchmark comparison table |

---

## 🖥️ CLI Inference

```bash
# Diagnose a single image
python predict.py --image data/processed/test/Healthy/Healthy_0001.jpg
```

---

## 🗂️ Project Structure

```
mint-leaf-ai/
├── backend/            FastAPI REST API server
│   └── app.py          Main server (POST /api/diagnose, GET /api/health)
├── frontend/           Web Dashboard
│   ├── index.html      Main UI
│   ├── style.css       Glassmorphic dark-mode styles
│   └── app.js          API integration & interactivity
├── inference/          ONNX Edge Inference Engine
│   └── engine.py       Fast CPU inference < 3.5 ms
├── xai/                Explainable AI
│   └── gradcam_service.py  Grad-CAM heatmap generator
├── utils/              Image Quality Pre-Screening
│   └── image_checker.py    Laplacian blur + exposure check
├── recommendation/     Agronomic Advisory Engine
│   └── advisory.py     Organic/chemical/cultural treatment DB
├── models/             Model Architectures
│   └── architectures/factory.py  25-model factory
├── training/           Training Pipeline
│   ├── trainers/trainer.py
│   ├── data/dataset.py
│   ├── losses/focal_loss.py
│   └── callbacks/callbacks.py
├── evaluation/         Evaluation Utilities
│   ├── metrics/evaluator.py
│   └── visualization/plotter.py
├── notebooks/          19 Research Notebooks (Steps 1-13)
├── data/               Dataset
│   ├── raw/            Original source images
│   ├── curated/        Manually curated images
│   └── processed/      Train/Val/Test splits (built by cleanup_and_build.py)
├── outputs/
│   ├── deployments/    ONNX + TorchScript model files
│   ├── experiments/    Per-model .pt checkpoints
│   ├── reports/        Benchmark CSVs, audit reports, research paper
│   └── visualizations/ Plots and charts
├── predict.py          Standalone CLI inference script
├── run_app.py          Web server launcher
├── check_startup.py    Pre-launch system diagnostic
├── requirements.txt    Python dependencies
└── scratch/            Development utility scripts
```

---

## 📊 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the web dashboard |
| `GET` | `/api/health` | System health + model status |
| `POST` | `/api/diagnose` | Upload image → full diagnosis pipeline |
| `GET` | `/api/benchmark` | 25-model leaderboard data |
| `GET` | `/docs` | Interactive Swagger API documentation |

---

## ⚠️ Important Caveats

1. **Test set size**: Internal evaluation on 313 images. External on 349 images. These are research-prototype scale, not large field trials.
2. **Grad-CAM**: Qualitative visual alignment with lesions — not quantitative IoU localization.
3. **Inference latency**: 3.45 ms measured on the development CPU. Hardware-specific.
4. **External generalization**: "Strong cross-source domain generalization under the evaluated external-source benchmark" — not universal real-world farm guarantee.

---

## 📄 Research Paper

Full research paper synthesis: `outputs/reports/final_paper/mint_leaf_ai_research_paper.md`

**Citation title:** *Mint Leaf AI: An Explainable, Deployment-Oriented Deep Learning System for Automated Mint Leaf Disease Recognition with Specimen-Aware Validation, Multi-Model Benchmarking, and ONNX Edge Deployment*

---

## 📦 Requirements

- Python 3.10+
- PyTorch 2.0+
- ONNX Runtime 1.16+
- FastAPI + Uvicorn
- OpenCV, Pillow, NumPy
- `timm` (for ConvNeXt/EfficientNet/Swin-T architectures)

See `requirements.txt` for exact versions.