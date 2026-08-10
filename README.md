# 🌿 Mint Leaf AI (Mint Leaf Classification & Diagnosis System)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Status](https://img.shields.io/badge/Project_Status-Step_2_Complete-brightgreen?style=for-the-badge)](#-14-stage-development-roadmap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An end-to-end AI-powered system for mint leaf classification, disease diagnosis, treatment recommendations, explainable AI (XAI) feature attribution, and real-time inference.

---

## 🗺️ 14-Stage Development Roadmap

| Stage | Module Name | Status | Description |
| :--- | :--- | :---: | :--- |
| **STEP 1** | **Project Foundation** | ✅ **COMPLETE** | Modular directory structure, environments, and configuration baseline. |
| **STEP 2** | **Dataset Audit Module** | ✅ **COMPLETE** | Automated non-destructive inspection (`01_dataset_audit.ipynb`), MD5 deduplication, and inventory generation. |
| **STEP 3** | **Taxonomy / Diagnosis Classes** | ⏳ UPCOMING | Class mapping, disease taxonomy, and standard target definitions. |
| **STEP 4** | **Dataset Cleaning & Organization** | ⏳ UPCOMING | Data normalization, deduplication execution, and structured directory organization. |
| **STEP 5** | **Preprocessing Pipeline** | ⏳ UPCOMING | Resolution standardization ($224\times224$), tensor transformations, and data loader setup. |
| **STEP 6** | **Model Framework** | ⏳ UPCOMING | Modular model registry & PyTorch base architecture setup. |
| **STEP 7** | **25 Models Training Suite** | ⏳ UPCOMING | Training 25 deep vision architectures (ResNet, EfficientNet, ConvNeXt, Vision Transformers, etc.) in Colab ML Lab. |
| **STEP 8** | **Model Comparison & Benchmark** | ⏳ UPCOMING | Metric aggregation (Accuracy, F1-score, Latency, Model Size) and leaderboards. |
| **STEP 9** | **Explainable AI (XAI)** | ⏳ UPCOMING | Grad-CAM, SHAP, and LIME interpretability visual maps. |
| **STEP 10** | **Diagnosis Engine** | ⏳ UPCOMING | Automated confidence scoring and multi-class disease diagnosis logic. |
| **STEP 11** | **RAG Recommendation Engine** | ⏳ UPCOMING | Retrieval-Augmented Generation for botanical treatment, pesticide, and care recommendations. |
| **STEP 12** | **Backend API** | ⏳ UPCOMING | FastAPI high-throughput REST API service for inference. |
| **STEP 13** | **Web Dashboard** | ⏳ UPCOMING | Interactive modern frontend UI for plant leaf uploads and diagnostic reporting. |
| **STEP 14** | **Full Integration** | ⏳ UPCOMING | End-to-end application deployment and validation. |

---

## 📂 Project Directory Structure

```text
mint-leaf-ai/
├── data/
│   ├── raw/               # Raw un-modified dataset folders
│   ├── processed/         # Cleaned, split, and transformed datasets
│   └── external/          # External reference data
├── notebooks/
│   └── 01_dataset_audit.ipynb  # Step 2: Automated dataset audit notebook
├── models/                # Trained weights and exported checkpoints (.pth, .onnx)
├── model_configs/         # Model architecture & hyperparameter YAML configs
├── backend/               # FastAPI backend REST service
├── frontend/              # Interactive Web Dashboard frontend
├── xai/                   # Grad-CAM, SHAP, and LIME interpretability scripts
├── recommendation/        # RAG recommendation engine for plant treatment
├── diagnosis/             # Disease classification and diagnostic logic
├── inference/             # Real-time model inference pipeline
├── utils/                 # Data loaders, image preprocessing, and loggers
├── outputs/
│   ├── predictions/       # JSON prediction outputs
│   ├── visualizations/    # Dataset audit dashboards & Grad-CAM heatmaps
│   └── reports/           # master_image_inventory.csv & dataset_audit_report.json
├── docs/                  # System architecture documentation
├── .agents/               # Workspace agent rules & git directives
├── requirements.txt       # Project dependencies
├── README.md              # Master project documentation
└── .gitignore             # Git ignore policies
```

---

## 🔬 Step 2: Dataset Audit Findings

The automated non-destructive audit executed in `01_dataset_audit.ipynb` yielded the following findings:

```text
DATASET AUDIT COMPLETE

Total Images: 4,031

Folders:
├── Mint leaf: 230
├── Mentha (Mint): 97
├── Fresh: 865
├── Spoiled: 300
├── Dried: 929
└── Augmented Mint Leaf: 1,610

Corrupted: 0
Exact Duplicates: 1,610

Most Common Resolution:
224 × 224

Reports Generated:
- outputs/reports/master_image_inventory.csv
- outputs/reports/dataset_audit_report.json
```

---

## 💻 Dual-Loop Architecture

```text
Antigravity IDE (Local Dev App/Backend/Frontend/XAI/RAG)
       │
  Git / Remote Sync
       │
       ▼
Google Colab (ML Laboratory: Audit ➔ Processing ➔ 25 Models Training ➔ Artifact Exports)
```

---

## 🚀 Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Praveen-K-0503/Mint_leaf_classification.git
   cd Mint_leaf_classification
   ```

2. **Set Up Python Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Dataset Audit Notebook**
   Launch VS Code or Jupyter Lab and execute `notebooks/01_dataset_audit.ipynb`.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.