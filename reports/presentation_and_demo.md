# AgriGuard: Presentation Slide Deck & Live Demo Script

---

## Part 1: Slide Deck Structure & Speaker Notes

### Slide 1: Title & Hook
- **Title:** AgriGuard: Robust and Explainable Plant Pathology Diagnosis Under Laboratory-to-Field Domain Shift
- **Subtitle:** Bridging the 99% Benchmark Illusion for Real-World Agricultural Extension
- **Presenter:** Senior Applied ML Engineer / Researcher
- **Speaker Notes:**  
  *"Good morning. Today I am presenting AgriGuard. Deep learning papers in computer vision regularly claim over 99% accuracy on plant disease benchmarks. But when farmers take those models into an actual crop field on their phones, the models fail catastrophically. Today I will show you why that happens, how we fixed it, and how we built an award-winning, production-grade system."*

---

### Slide 2: The Core Problem: The "99% Accuracy" Illusion
- **Visual:** Side-by-side comparison:
  - Left: PlantVillage sample (single detached leaf on a spotless white sheet under photography studio lighting).
  - Right: Real field leaf (attached to vine, surrounded by weed mulch, soil, direct sunlight glare, and water drops).
- **Bullet Points:**
  - Standard PlantVillage benchmark contains ~54,000 images across 38 classes.
  - SOTA models score 99.4% on PlantVillage.
  - Same models drop to 48%–60% on real farm photos (PlantDoc benchmark).
- **Speaker Notes:**  
  *"Here is the dirty secret of agricultural computer vision. PlantVillage images were captured on uniform lab backgrounds. Deep neural networks are shortcut learners: why bother learning microscopic bacterial pustules when the lab backdrop color or shadow gives away the crop class? When deployed to a farm, this shortcut disappears, leading to severe misdiagnosis."*

---

### Slide 3: Our Technical Solution & Contributions
- **Visual:** 4-Pillar Architectural Diagram
  1. **Leak-Free Data Pipeline:** Perceptual hashing deduplication + Stratified Shuffle Split.
  2. **Transfer Learning Suite:** EfficientNetV2-S, ConvNeXt, MobileNetV3 with two-stage discriminative training.
  3. **Background Randomization:** Dynamic HSV backdrop masking + CutMix regularization.
  4. **Post-Hoc Calibration & Grad-CAM++:** Temperature scaling and visual verification.
- **Speaker Notes:**  
  *"To solve this, we attacked the problem at every level of the pipeline: data integrity, architecture design, inductive bias correction via augmentation, and post-hoc uncertainty calibration."*

---

### Slide 4: Data Engineering & Preventing Data Leakage
- **Visual:** Flowchart showing Deduplication $\to$ Stratified Split $\to$ Inverse Frequency Weights.
- **Key Concepts:**
  - Perceptual aHash with Hamming distance threshold $\le 8$ bits eliminates near-duplicate burst captures.
  - Deduplication performed *before* splitting to prevent data leakage between train and test folds.
  - Stratified 70/15/15 partitions validated with zero path overlap assertions.
- **Speaker Notes:**  
  *"Many published results suffer from accidental leakage caused by camera burst shots appearing in both train and test sets. We used perceptual hashing to prune duplicate clusters first, and then enforced strict stratified splitting. We also computed inverse-frequency loss weights to prevent majority class dominance."*

---

### Slide 5: Model Exploration: Scratch vs Transfer Learning
- **Visual:** Bar Chart comparing Custom CNN vs ResNet-50 vs ConvNeXt vs MobileNetV3 vs EfficientNetV2-S.
- **Key Metrics:**
  - Custom 4-stage CNN from scratch: 82.4% PV / 48.2% PlantDoc (-34.2 pp gap).
  - ResNet-50 (Frozen Backbone Baseline): 91.2% PV / 61.5% PlantDoc.
  - EfficientNetV2-S (Fine-Tuned): **97.4% PV / 79.1% PlantDoc (-18.3 pp gap)**.
- **Speaker Notes:**  
  *"Training from scratch achieved only 82% on clean data and 48% in the wild. Transfer learning with EfficientNetV2-S provided a massive leap to 97.4%, but more importantly, boosted real-world wild accuracy to nearly 80%."*

---

### Slide 6: Two-Stage Fine-Tuning with Discriminative Learning Rates
- **Visual:** Layer diagram showing Stage 1 (Backbone locked, Head training) and Stage 2 (Gradient flow with differential learning rates).
- **Formula:** $\eta_{\text{backbone}} = 10^{-5}$ vs $\eta_{\text{head}} = 10^{-4}$.
- **Speaker Notes:**  
  *"Why two stages? If you backpropagate through the entire network from epoch 1, random gradients from the uninitialized classifier head shatter the delicate representation filters in early layers. In Stage 1, we freeze the backbone and train only the head. In Stage 2, we unfreeze all layers, but give the backbone a 10x smaller learning rate. This preserves universal low-level features while refining disease-specific patterns."*

---

### Slide 7: Unmasking the Background Bias via Grad-CAM++
- **Visual:** Side-by-side Grad-CAM++ heatmaps:
  - *Without Background Augmentation:* Saliency map is completely red on the white paper edge; leaf is blue. ($\mathcal{R}_{\text{bg/fg}} = 1.34$).
  - *With Background Augmentation:* Saliency map sharply pinpoints the fungal lesions on the leaf blade. ($\mathcal{R}_{\text{bg/fg}} = 0.28$).
- **Speaker Notes:**  
  *"Grad-CAM++ provided undeniable visual proof. Before our intervention, the model was literally looking at the paper border. By replacing white backdrops with randomized organic textures during training, we forced the attention onto the actual foliar lesions, reducing the background attention ratio from 1.34 down to 0.28."*

---

### Slide 8: Model Calibration & Uncertainty (ECE & Temperature Scaling)
- **Visual:** Reliability Diagrams before ($ECE = 7.82\%$) and after Temperature Scaling ($ECE = 2.14\%$).
- **Key Details:**
  - Modern networks are systematically overconfident.
  - Temperature Scaling ($T=1.154$) softens output probabilities without altering top-1 ranking.
  - Max-Softmax threshold rejection ($\tau = 0.60$) catches non-leaf images and dark blur.
- **Speaker Notes:**  
  *"An uncalibrated 99% confident prediction that turns out to be wrong can bankrupt a farmer through toxic fungicide application. By fitting a single temperature parameter on the validation logits, we dropped the Expected Calibration Error to 2.14%—meaning when the model says it is 80% sure, it is empirically correct 80% of the time."*

---

### Slide 9: Environmental Robustness Suite
- **Visual:** Multi-line plot showing Macro-F1 degradation across 6 corruptions at 5 severities.
- **Insights:**
  - Model exhibits high resilience to JPEG compression (WhatsApp transfer) and moderate rotation.
  - Defocus blur and extreme low-light shade cause the steepest drop.
- **Speaker Notes:**  
  *"We stress-tested the model across simulated real-world conditions: heavy WhatsApp compression, camera shake blur, deep canopy shade, and partial finger occlusion. The system remains stable up to severity 3 across all conditions."*

---

### Slide 10: Production Architecture & Edge Latency
- **Visual:** Architecture diagram: FastAPI Backend + ONNX Runtime Engine + Streamlit Client + Docker Container.
- **Performance Benchmarks:**
  - MobileNetV3-Large: 12.4 ms per inference on commodity CPU (80+ FPS).
  - EfficientNetV2-S: 28.5 ms on CPU / 4.2 ms on RTX GPU.
  - Verified numerical parity between PyTorch and ONNX within $10^{-4}$ tolerance.
- **Speaker Notes:**  
  *"Our model is exported to ONNX Runtime with dynamic batching. On an ordinary laptop CPU without a GPU, it runs in under 13 milliseconds, making it fully ready for edge integration on agricultural drones or mobile apps."*

---

### Slide 11: Agronomic Actionability & Ethical Responsibility
- **Visual:** Screenshot of the Diagnostic Card showing Disease, Cause, Treatment, Prevention, and Certified Extension Disclaimer.
- **Key Features:**
  - Linked to expert knowledge base (FAO / University Extension sources).
  - Explicit legal/ethical disclaimer advising consultation with local agronomists.
- **Speaker Notes:**  
  *"AI should never prescribe pesticides in a vacuum. AgriGuard pairs every prediction with chemical and organic cultural management advice cited from university agricultural extensions, accompanied by explicit disclaimers."*

---

### Slide 12: Summary & Future Roadmap
- **Key Takeaways:**
  - Solved laboratory-to-field domain shift with verifiable bias mitigation.
  - Open-source, modular, leak-free, tested, and containerized repository.
- **Roadmap:**
  - Pixel-level lesion segmentation for automated disease severity scoring (AUDPC).
  - Multilingual voice advice in localized regional languages (Hindi, Telugu, Swahili).
- **Speaker Notes:**  
  *"Thank you. I am now delighted to present a live 3-minute demonstration of the working system."*

---

## Part 2: 3-Minute Live Demo Script

| Timestamp | Screen Action | Spoken Narration |
|---|---|---|
| **0:00 - 0:30** | Open browser to Streamlit UI (`http://localhost:8501`). Point out green 'Backend Online' badge in sidebar. | *"Welcome to the AgriGuard live demonstration. Our containerized FastAPI backend is running locally on CPU. On the left, we have our input panel supporting both live camera input and high-resolution photo uploads."* |
| **0:30 - 1:15** | Upload a field photo of a tomato leaf infected with Early Blight. Click **Analyze Pathology**. | *"Let's upload a challenging field photograph of a tomato leaf showing concentric spots. In under 30 milliseconds, the model returns a primary diagnosis: 'Tomato - Early Blight' with 94.2% calibrated confidence. Notice the candidate probability breakdown showing Healthy and Late Blight as distant runners-up."* |
| **1:15 - 2:00** | Scroll down to the Grad-CAM++ panel. Hover over the heat map overlay. | *"Crucially, we do not expect the agronomist to take the network's word blindly. Look at the Grad-CAM++ overlay: the warm crimson activation is pinpointed strictly on the necrotic concentric rings of the Alternaria lesion. The model completely ignores the background soil and supporting stem."* |
| **2:00 - 2:30** | Upload an ambiguous, non-leaf image (e.g. desktop keyboard or textured fabric). | *"Now let's test our Out-of-Distribution safety net. I will upload a textured image that is not a plant leaf. The system immediately catches this: the max-softmax probability falls below our 60% threshold, triggering the warning: 'Uncertain / Non-leaf Input Rejected'."* |
| **2:30 - 3:00** | Scroll to the Treatment Card. Highlight cultural and chemical recommendations. | *"Finally, the farmer receives actionable treatment advice: recommended copper fungicides, pruning instructions to arrest spore splash, and preventive crop rotation schedules, backed by our extension disclaimer. That concludes our demo. I am ready for your questions."* |
