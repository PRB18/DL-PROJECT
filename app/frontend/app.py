"""
app/frontend/app.py
-------------------
Interactive Streamlit application for Farmers and Field Agronomists:
- Live Camera snapshot capture or direct Photo Upload.
- Real-time communication with FastAPI backend.
- Displays Top-3 diagnostic predictions with calibrated confidence meters.
- Interactive Grad-CAM++ lesion localization overlay.
- Actionable agronomic treatment, prevention protocols, and certified disclaimers.
"""

import base64
import io
import json
import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AgriGuard | Plant Disease Diagnostic System",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 AgriGuard: Intelligent Plant Disease Diagnostic Suite")
st.markdown(
    """
    **AI-Assisted Field Crop Pathology & Treatment Assistant**  
    *Powered by EfficientNetV2-S Transfer Learning with Grad-CAM++ Explainability.*
    """
)

# Sidebar System Health & Info
with st.sidebar:
    st.header("⚙️ System Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success(f"Backend Online ({r.json().get('device', 'CPU')})")
        else:
            st.warning("Backend degraded")
    except Exception:
        st.error("Backend Offline. Start `uvicorn app.backend.main:app`")

    st.markdown("---")
    st.markdown("### 🌾 Target Crops")
    st.markdown("- Tomato, Potato, Bell Pepper\n- Apple, Grape, Corn (Maize)\n- Strawberry, Peach, Cherry, Squash")
    st.markdown("---")
    st.info("💡 **Tip**: Hold the camera ~15-20cm from the foliage. Ensure the affected spot is well-lit.")

# Main layout: Upload / Camera section
col_input1, col_input2 = st.columns([1, 1])

with col_input1:
    st.subheader("1. Provide Foliage Photo")
    input_mode = st.radio("Choose Input Mode:", ["Upload Image File", "Use Device Camera"], horizontal=True)
    uploaded_file = None

    if input_mode == "Upload Image File":
        uploaded_file = st.file_uploader("Upload leaf photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Take a photo of the affected plant leaf")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Specimen", use_column_width=True)

with col_input2:
    st.subheader("2. AI Diagnosis & Analysis")
    if uploaded_file is not None:
        if st.button("🔍 Analyze Pathology", type="primary", use_container_width=True):
            with st.spinner("Executing neural inference and generating visual explanation..."):
                try:
                    # Send image to FastAPI
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format="JPEG")
                    files = {"file": ("query.jpg", img_bytes.getvalue(), "image/jpeg")}

                    resp = requests.post(f"{API_URL}/predict", files=files, timeout=20)
                    if resp.status_code == 200:
                        data = resp.json()
                        top_pred = data["top_predictions"][0]

                        # Check uncertainty
                        if data.get("is_uncertain_or_non_leaf", False):
                            st.warning(
                                f"⚠️ **Low Confidence Warning** (Top confidence {top_pred['probability']*100:.1f}%). "
                                "The image may not contain a recognized leaf, or the lighting/blur is severe."
                            )

                        # Primary Diagnosis Card
                        st.success(f"### Primary Diagnosis: **{top_pred['display_name']}**")
                        st.metric(label="Calibrated Confidence", value=f"{top_pred['probability']*100:.1f}%")

                        # Top-3 Probabilities
                        st.markdown("#### Candidate Probabilities:")
                        for p in data["top_predictions"]:
                            st.write(f"**{p['display_name']}**")
                            st.progress(min(1.0, max(0.0, p["probability"])))

                        # Grad-CAM Visual Overlay
                        st.markdown("---")
                        st.markdown("#### 🔬 Visual Verification (Grad-CAM++)")
                        st.caption("Warm colors (Red/Yellow) highlight areas that triggered the disease classification.")
                        if data.get("heatmap_base64"):
                            heatmap_bytes = base64.b64decode(data["heatmap_base64"])
                            st.image(heatmap_bytes, caption="Grad-CAM++ Lesion Localization", use_column_width=True)

                        # Actionable Treatment & Advice Card
                        st.markdown("---")
                        st.markdown("### 📋 Agronomic Treatment & Management")
                        advice = data.get("treatment_advice", {})
                        if advice:
                            st.markdown(f"**Disease/Pathogen:** {advice.get('disease', 'N/A')}")
                            st.markdown(f"**Symptoms:** {advice.get('symptoms', 'N/A')}")
                            st.info(f"**Recommended Treatment:**\n{advice.get('treatment', 'N/A')}")
                            st.success(f"**Prevention Strategy:**\n{advice.get('prevention', 'N/A')}")
                            st.caption(f"Source Authority: {advice.get('source', 'Agricultural Research Center')}")

                        # Agronomist Disclaimer
                        st.warning(f"⚖️ {data.get('disclaimer', '')}")

                    else:
                        st.error(f"Prediction failed with status {resp.status_code}: {resp.text}")
                except Exception as ex:
                    st.error(f"Inference error: {ex}")
    else:
        st.info("👈 Upload or capture a leaf photo to begin diagnosis.")
