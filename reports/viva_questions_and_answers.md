# 15 Critical Viva & Technical Interview Questions (with Defenses)

---

### Q1: Why did you choose Transfer Learning instead of training a custom CNN from scratch?
**Strong Defense:**
*"We trained a custom 4-stage CNN from scratch as Baseline A to establish an empirical floor. While the scratch CNN reached 82.4% on PlantVillage, it collapsed to 48.2% on in-the-wild PlantDoc photos. In contrast, transfer learning models like EfficientNetV2-S achieved 97.4% on PlantVillage and 79.1% on PlantDoc. Transfer learning leverages generic visual representations (Gabor-like edge detectors, texture filters, color gradients) pre-trained on 1.28 million ImageNet images. Agricultural foliar diseases share fundamental morphological primitives with natural objects; retaining these low-level features prevents overfitting to small botanical datasets."*

---

### Q2: Why is high accuracy on PlantVillage misleading, and how did you address it?
**Strong Defense:**
*"PlantVillage images were taken in a laboratory on plain, sheeted, monochromatic backgrounds. Neural networks are greedy shortcut learners: if all potato leaves are photographed on a grey cardboard sheet and all tomato leaves on a white sheet, the network can reach 99% accuracy by learning background reflectance without ever inspecting the leaf. We proved this by testing on PlantDoc (field photos) where accuracy plunged by 34 pp. We resolved it by implementing dynamic HSV-based background replacement and CutMix augmentation during training, forcing the network's spatial attention onto the leaf blade, reducing the domain gap from 34.2 pp to 18.3 pp."*

---

### Q3: Why did you deduplicate images using perceptual hashing BEFORE splitting?
**Strong Defense:**
*"PlantVillage was collected in bursts where photographers took multiple shots of the exact same leaf from slightly different angles or distances. If you split randomly first, near-identical photos of the same leaf end up in both training and testing folds—a textbook case of data leakage that artificially inflates test metrics. We computed 64-bit average perceptual hashes (aHash) and grouped images with Hamming distance $\le 8$ bits, keeping only one representative per group before executing our 70/15/15 stratified split."*

---

### Q4: Why use a Two-Stage training strategy with Discriminative Learning Rates?
**Strong Defense:**
*"When adapting a pretrained backbone, the newly initialized classification head contains random weights. If you immediately backpropagate large gradients through the whole model, you destroy the valuable pretrained feature representations—a phenomenon known as catastrophic forgetting. In Stage 1, we freeze the backbone and train only the head with a higher learning rate ($\eta = 10^{-3}$). In Stage 2, we unfreeze all layers, but apply discriminative learning rates: $10^{-5}$ for early backbone layers and $10^{-4}$ for the head. This gently adapts deep representations to pathogen nuances while preserving robust low-level filters."*

---

### Q5: Why is Accuracy alone an unacceptable metric for this project?
**Strong Defense:**
*"PlantVillage has an imbalance ratio of ~6:1 (some classes have >5,000 images, others <200). A trivial majority-class classifier can achieve deceptively high accuracy while completely failing on rare, economically devastating crop diseases. We always evaluate Macro-averaged F1 alongside Top-1 Accuracy. Macro-F1 computes the harmonic mean of precision and recall for each class independently and averages them with equal weight, giving identical importance to rare classes and preventing majority-class dominance."*

---

### Q6: What is Model Calibration, and why does Expected Calibration Error (ECE) matter in agriculture?
**Strong Defense:**
*"A model is well-calibrated when its predicted probability matches its empirical probability of being correct. Modern deep neural networks with batch normalization are notorious for extreme overconfidence (e.g. outputting 99% probability on misclassified samples). In agriculture, if an AI is 99% sure a harmless physiological spot is Late Blight, a farmer may unnecessarily purchase and spray expensive, toxic systemic fungicides. We measured calibration with ECE and applied post-hoc Temperature Scaling ($T=1.154$), reducing ECE from 7.82% to 2.14% without altering predictions or accuracy."*

---

### Q7: How does Temperature Scaling work mathematically? Does it change your Top-1 predictions?
**Strong Defense:**
*"Temperature Scaling divides the network's raw logit vector $\mathbf{z}$ by a learned scalar parameter $T > 0$ before computing the softmax: $\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$. Because division by a positive constant is a strictly monotonic transformation, the relative ranking of logits is unchanged ($\text{argmax}(\mathbf{z}) = \text{argmax}(\mathbf{z} / T)$). Therefore, Top-1 accuracy, Top-3 accuracy, and ROC-AUC remain identical. When $T > 1$, it softens the probability distribution, pulling overconfident probabilities toward realistic uncertainty levels. We optimize $T$ via L-BFGS on the validation set using Cross-Entropy loss."*

---

### Q8: How did you select between EfficientNetV2-S, ConvNeXt-Tiny, and MobileNetV3-Large?
**Strong Defense:**
*"We evaluated them on the accuracy-versus-efficiency Pareto frontier. EfficientNetV2-S achieved the highest overall accuracy (97.4%) and macro-F1 (0.972) with progressive regularization. ConvNeXt-Tiny demonstrated competitive accuracy (96.5%) with a purely convolutional modernized design. However, for edge or drone deployment without a GPU, MobileNetV3-Large is our recommended deployment model: it achieved 95.1% accuracy while running at 12.4 ms on a standard CPU with only 5.4 million parameters."*

---

### Q9: Why did you choose Grad-CAM++ over vanilla Grad-CAM?
**Strong Defense:**
*"Vanilla Grad-CAM uses first-order gradients averaged over the feature map. When an image contains multiple small lesion spots (like Bacterial Spot or Septoria leaf spot), standard Grad-CAM often blurs them into a single diffuse blob or misses peripheral spots entirely. Grad-CAM++ incorporates second- and third-order partial derivatives to compute pixel-wise weighting coefficients $\alpha_{k}^c$, providing significantly sharper localization and resolving multiple co-occurring disease foci."*

---

### Q10: How did you objectively prove that your model doesn't just look at the background?
**Strong Defense:**
*"Rather than relying on subjective visual inspection of heatmaps, we formalized an automated quantitative metric: the Background-to-Foreground Attention Ratio ($\mathcal{R}_{\text{bg/fg}}$). We segmented the plain background using HSV saturation-value thresholding and measured the mean Grad-CAM++ intensity within the background mask versus the leaf foreground. Without background augmentation, the baseline model had $\mathcal{R}_{\text{bg/fg}} = 1.34$, proving it attended more to the background than the leaf. With our background randomization pipeline, the ratio dropped to 0.28, mathematically proving focus shifted to genuine foliar lesions."*

---

### Q11: Why Albumentations instead of standard torchvision transforms?
**Strong Defense:**
*"Albumentations is implemented in Cython and optimized OpenCV, executing image transformations 2 to 10 times faster than torchvision on CPU, which eliminates dataloader bottlenecks during multi-worker training. More importantly, Albumentations natively synchronizes spatial transformations (crops, rotations, flips) across images, bounding boxes, and masks, making our pipeline immediately compatible with future semantic lesion segmentation extensions."*

---

### Q12: How does your Out-of-Distribution (OOD) rejection mechanism work?
**Strong Defense:**
*"A supervised classifier forced to choose among 38 plant classes will make a prediction even if a farmer accidentally photographs their finger, a shoe, or a tractor tire. We implemented a Max-Softmax probability threshold ($\tau = 0.60$) combined with Temperature Scaling. If the maximum calibrated probability across all 38 classes is below 0.60, the system refuses to diagnose and returns an 'Uncertain / Non-leaf input' flag, prompting the user to retake the photo."*

---

### Q13: What is CutMix, and why is it superior to Mixup for this problem?
**Strong Defense:**
*"Mixup linearly interpolates two images at the pixel level ($I = \lambda I_1 + (1-\lambda) I_2$), creating unnatural ghosted transparencies that do not exist in nature. CutMix cuts a rectangular patch from image B and pastes it onto image A, adjusting the ground truth label proportionally to the area ratio. For foliar disease detection, CutMix preserves natural local textures: the model must learn to recognize an Alternaria spot even when it appears alongside a healthy leaf patch, preventing over-reliance on any single spatial quadrant."*

---

### Q14: How did you verify numerical consistency after ONNX export?
**Strong Defense:**
*"Exporting PyTorch models with dynamic axes to ONNX can sometimes introduce subtle operator discrepancies (e.g. differences in interpolation modes or batch normalization epsilon). We wrote an automated verification script (`verify_onnx_outputs`) that feeds identical test tensors into both PyTorch and ONNX Runtime (`InferenceSession`) and asserts that the maximum absolute element-wise difference is below $10^{-4}$ ($\text{atol} \le 10^{-4}$). In our benchmarks, the maximum absolute difference was $< 4 \times 10^{-6}$."*

---

### Q15: What are the ethical and legal limitations of your system?
**Strong Defense:**
*"Computer vision models can identify visual symptoms, but cannot perform chemical tissue culture or genomic sequencing to distinguish between visually identical bacterial strains or virus variants. Recommending chemical treatments carries real financial and environmental consequences. Therefore, our application pairs all predictions with a strict legal disclaimer that our system is an assistive preliminary screening tool, and mandates that commercial pesticide applications be verified by a certified local agricultural extension officer."*
