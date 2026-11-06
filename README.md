# Plant Disease Detection 🌿🔍

> **Classify plant leaf diseases from photos using CNNs and transfer learning — built for real-world farm conditions, not just lab benchmarks.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## The Problem with "99% Accuracy"

Most plant disease papers report near-perfect accuracy on PlantVillage — a dataset where leaves are photographed on **plain white/grey backgrounds in controlled lab lighting**. 

A model trained this way **does not generalise to real farm photos**. It learns "white background → healthy" rather than "necrotic lesion pattern → disease". 

We address this honestly:
- Train on PlantVillage (38 classes, ~54 k images)  
- Test also on **PlantDoc** (in-the-wild, real farm photos)  
- Report the **domain gap** explicitly  
- Fix it with background-randomisation augmentation and show before/after Grad-CAM

---

## Results Summary

> *Numbers below are illustrative placeholders. Run the training pipeline and update this table.*

| Model | PV Test Acc | PV Macro F1 | PlantDoc Acc | PlantDoc F1 | Params | Latency (CPU) |
|---|---|---|---|---|---|---|
| Baseline CNN | ~82% | ~0.79 | ~48% | ~0.44 | 1.2M | 18 ms |
| ResNet50 (frozen) | ~91% | ~0.90 | ~61% | ~0.58 | 23.5M | 45 ms |
| EfficientNetV2-S ✅ | ~97% | ~0.97 | ~79% | ~0.76 | 21.5M | 28 ms |
| ConvNeXt-Tiny | ~96% | ~0.96 | ~77% | ~0.74 | 28.6M | 32 ms |
| MobileNetV3 (deploy) | ~95% | ~0.94 | ~74% | ~0.71 | 5.4M | 12 ms |

**Domain gap before/after background augmentation:** −18 pp → −8 pp (EfficientNetV2-S)

---

## Repository Structure

```
plant-disease-detection/
├── configs/
│   └── config.yaml              ← All hyperparameters live here
├── src/
│   ├── data/
│   │   ├── download.py          ← Dataset download (Kaggle + GitHub)
│   │   ├── eda.py               ← Exploratory data analysis
│   │   ├── splits.py            ← Leak-free stratified splits
│   │   ├── dataset.py           ← PyTorch Dataset + Albumentations
│   │   ├── class_weights.py     ← Inverse-frequency class weights
│   │   └── domain_shift.py      ← Domain-shift evaluation
│   ├── models/
│   │   ├── baseline_cnn.py      ← Small CNN from scratch
│   │   └── transfer.py          ← timm model factory
│   ├── training/
│   │   ├── trainer.py           ← Training loop (AMP, early stopping)
│   │   ├── losses.py            ← CE + CutMix loss
│   │   └── scheduler.py        ← Cosine LR with warmup
│   ├── evaluation/
│   │   ├── metrics.py           ← All metrics (F1, AUC, ECE, etc.)
│   │   ├── calibration.py       ← Temperature scaling
│   │   ├── robustness.py        ← Corruption tests
│   │   └── onnx_export.py       ← ONNX export + verification
│   ├── explainability/
│   │   └── gradcam.py           ← Grad-CAM++ overlays
│   └── utils/
│       ├── seed.py              ← Global seed setter
│       ├── config.py            ← OmegaConf loader
│       └── logger.py            ← Console + TensorBoard logger
├── app/
│   ├── backend/
│   │   ├── main.py              ← FastAPI app (/predict, /health)
│   │   └── knowledge_base.json  ← Disease info + treatment tips
│   └── frontend/                ← React / Streamlit UI
├── notebooks/
│   ├── 01_eda_and_splits.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_explainability.ipynb
│   └── 06_demo.ipynb
├── reports/
│   ├── figures/                 ← All EDA and results plots
│   └── technical_report.pdf    ← 6-8 page IEEE-style report
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── checkpoints/                 ← Saved model weights (.pth + .onnx)
├── data/
│   ├── raw/                     ← Downloaded datasets (not in git)
│   ├── processed/
│   └── splits/                  ← train/val/test CSVs
├── requirements.txt
├── Dockerfile                   ← Backend container
├── docker-compose.yml
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download datasets
```bash
# PlantVillage (requires Kaggle API key)
# See: https://www.kaggle.com/datasets/emmarex/plantdisease
python src/data/download.py --plantvillage

# PlantDoc (free, no account needed)
python src/data/download.py --plantdoc
```

### 3. Run EDA and build splits
```bash
python src/data/eda.py
python src/data/splits.py
```

### 4. Train
```bash
# Phase 2: Baselines
python src/training/train_baseline.py

# Phase 3: Transfer learning (best model)
python src/training/train_transfer.py --model efficientnetv2_s
```

### 5. Evaluate + explain
```bash
python src/evaluation/evaluate.py --checkpoint checkpoints/best_efficientnetv2_s.pth
python src/explainability/gradcam_report.py
```

### 6. Run the app
```bash
# Backend
uvicorn app.backend.main:app --reload

# Frontend (Streamlit)
streamlit run app/frontend/app.py

# Or Docker Compose
docker-compose up
```

---

## Design Decisions (for Viva)

| Decision | Choice | Why |
|---|---|---|
| Augmentation library | Albumentations | 2–10× faster than torchvision; supports mask-consistent spatial augmentations for future segmentation |
| Backbone family | EfficientNetV2-S, ConvNeXt-Tiny, MobileNetV3 | Covers accuracy-efficiency tradeoff; all available in timm |
| Training strategy | Two-stage (freeze → unfreeze with discriminative LRs) | Prevents catastrophic forgetting of ImageNet features; stage 1 warms up the head, stage 2 fine-tunes everything |
| Loss | CrossEntropy + label smoothing 0.1 | Label smoothing prevents overconfident predictions; helps calibration |
| Class imbalance | Weighted loss vs balanced sampler (ablation) | Both are tested; weighted loss is simpler and usually comparable |
| Evaluation | Accuracy + macro-F1 + per-class report | Accuracy alone is misleading on imbalanced data; macro-F1 weights all classes equally |
| Explainability | Grad-CAM++ | Sharper than vanilla Grad-CAM; works with any CNN without modifying architecture |
| OOD rejection | Max-softmax threshold | Simple, fast, works at inference time without retraining |

---

## Datasets

| Dataset | Images | Classes | Background | Use |
|---|---|---|---|---|
| PlantVillage | ~54,000 | 38 | Plain / lab-style | Train + Val + Test |
| PlantDoc | ~2,600 | 27 | In-the-wild / farm | Domain-shift test only |

---

## Limitations

1. **Domain gap**: Even after background augmentation, there is still a ~8–12 pp gap between lab and real-world accuracy. Field photos vary in lighting, angle, and partial occlusion in ways training data doesn't cover.
2. **Disease severity**: We classify disease *presence* but not *severity* (early vs late stage). This is flagged as future work.
3. **Regional bias**: PlantVillage is a North American dataset. Disease variants and crop cultivars differ by region (India, Africa, South-East Asia).
4. **Disclaimer**: This tool is for educational and preliminary screening only. **Always consult a qualified agronomist for treatment decisions.**

---

## Citation

```bibtex
@misc{plantdisease2024,
  title  = {Plant Disease Detection with Transfer Learning and Domain-Shift Evaluation},
  author = {Your Name},
  year   = {2024},
  note   = {GitHub repository}
}
```

---

## Acknowledgements

- PlantVillage dataset: Hughes & Salathé (2015), arXiv:1511.08060  
- PlantDoc dataset: Singh et al. (2020), CODS-COMAD '20  
- timm library: Ross Wightman  
- pytorch-grad-cam: Jacob Gildenblat  
