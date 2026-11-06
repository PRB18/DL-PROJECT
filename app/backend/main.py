"""
app/backend/main.py
-------------------
FastAPI production inference service for Plant Disease Detection:
- /health: System readiness and model check.
- /classes: List of recognized 38 plant disease categories.
- /predict: Multipart image upload returning Top-3 predictions, calibrated confidence,
  out-of-distribution uncertainty flag, base64 Grad-CAM++ diagnostic overlay, and
  actionable agronomic treatment advice.
"""

import base64
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data.dataset import get_val_transforms
from src.explainability.gradcam import GradCAMPlusPlus, get_target_layer, overlay_heatmap
from src.models.transfer import build_model
from src.utils.config import load_config

app = FastAPI(
    title="Plant Disease Detection API",
    version="1.0.0",
    description="Deep learning inference server providing plant pathology diagnosis and explainability.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables loaded on startup
MODEL: Optional[torch.nn.Module] = None
TARGET_LAYER: Optional[torch.nn.Module] = None
CLASS_NAMES: List[str] = []
KNOWLEDGE_BASE: Dict = {}
CFG = None
TRANSFORM = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OOD_THRESHOLD = 0.60
TEMPERATURE = 1.15  # Default post-hoc temperature scaling factor


class PredictionItem(BaseModel):
    class_name: str
    display_name: str
    probability: float


class PredictResponse(BaseModel):
    top_predictions: List[PredictionItem]
    is_uncertain_or_non_leaf: bool
    uncertainty_score: float
    treatment_advice: Dict[str, str]
    disclaimer: str
    heatmap_base64: str


@app.on_event("startup")
def load_artifacts():
    global MODEL, TARGET_LAYER, CLASS_NAMES, KNOWLEDGE_BASE, CFG, TRANSFORM, OOD_THRESHOLD

    cfg_path = Path("configs/config.yaml")
    if cfg_path.exists():
        CFG = load_config(cfg_path)
    else:
        # Fallback dummy config
        from omegaconf import OmegaConf
        CFG = OmegaConf.create({"data": {"num_classes": 38, "image_size": 224}})

    TRANSFORM = get_val_transforms(CFG.data.image_size)

    # Load Knowledge Base
    kb_path = Path("app/backend/knowledge_base.json")
    if kb_path.exists():
        with open(kb_path, "r", encoding="utf-8") as f:
            KNOWLEDGE_BASE = json.load(f)
    else:
        KNOWLEDGE_BASE = {"disclaimer": "Consult local agronomist.", "diseases": {}, "default": {}}

    # Load class mapping
    class_map_path = Path("data/splits/plantvillage_class_map.json")
    if class_map_path.exists():
        with open(class_map_path, "r") as f:
            data = json.load(f)
            idx_to_class = data["idx_to_class"]
            CLASS_NAMES = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
    else:
        CLASS_NAMES = [f"Class_{i}" for i in range(CFG.data.num_classes)]

    # Load model
    model_path = Path("checkpoints/best_efficientnetv2_s.pth")
    MODEL = build_model("efficientnetv2_s", num_classes=len(CLASS_NAMES), pretrained=False)
    if model_path.exists():
        state = torch.load(model_path, map_location=DEVICE)
        MODEL.load_state_dict(state.get("state_dict", state))
        print(f"Loaded trained weights from {model_path}")
    else:
        print("Using initialized model weights (training checkpoint not found).")

    MODEL.to(DEVICE)
    MODEL.eval()

    try:
        TARGET_LAYER = get_target_layer(MODEL, "efficientnetv2_s")
    except Exception as e:
        print(f"Warning targeting layer: {e}")
        TARGET_LAYER = None


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "num_classes": len(CLASS_NAMES),
        "model_loaded": MODEL is not None,
    }


@app.get("/classes")
def get_classes():
    return {"classes": CLASS_NAMES}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")

    content = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted image file.")

    img_rgb = np.array(pil_img)
    augmented = TRANSFORM(image=img_rgb)
    tensor_input = augmented["image"].unsqueeze(0).to(DEVICE)

    # 1. Forward Pass with Temperature Scaling
    with torch.no_grad():
        logits = MODEL(tensor_input)
        scaled_logits = logits / TEMPERATURE
        probs = F.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy()

    # 2. Extract Top-3
    top3_indices = np.argsort(probs)[-3:][::-1]
    top_preds = []
    for idx in top3_indices:
        raw_name = CLASS_NAMES[idx]
        formatted_name = raw_name.replace("___", " - ").replace("_", " ")
        top_preds.append(
            PredictionItem(
                class_name=raw_name,
                display_name=formatted_name,
                probability=float(probs[idx]),
            )
        )

    top_prob = float(probs[top3_indices[0]])
    is_uncertain = bool(top_prob < OOD_THRESHOLD)
    uncertainty_score = float(1.0 - top_prob)

    # 3. Grad-CAM++ Explanation for Top-1 class
    heatmap_base64 = ""
    if TARGET_LAYER is not None:
        try:
            cam_engine = GradCAMPlusPlus(MODEL, TARGET_LAYER)
            cam = cam_engine.generate(tensor_input, target_class=int(top3_indices[0]))
            cam_engine.remove_hooks()

            overlay_rgb = overlay_heatmap(img_rgb, cam, alpha=0.45)
            # Encode overlay as base64 JPEG
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            heatmap_base64 = base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            print(f"Grad-CAM generation error: {e}")

    # 4. Fetch Treatment from Knowledge Base
    top_class_key = CLASS_NAMES[top3_indices[0]]
    disease_info = KNOWLEDGE_BASE.get("diseases", {}).get(
        top_class_key, KNOWLEDGE_BASE.get("default", {})
    )

    return PredictResponse(
        top_predictions=top_preds,
        is_uncertain_or_non_leaf=is_uncertain,
        uncertainty_score=uncertainty_score,
        treatment_advice=disease_info,
        disclaimer=KNOWLEDGE_BASE.get("disclaimer", "Consult a certified agronomist."),
        heatmap_base64=heatmap_base64,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=True)
