

#  Hierarchical Leaf–Disease Segmentation

A Unified ConvNeXt-Tiny Framework for Plant Health Monitoring

This repository contains a complete implementation of a **hierarchical panoptic segmentation model** capable of detecting **leaf regions** and **fine-grained disease lesions** using a dual-headed architecture. It combines the strengths of **ConvNeXt-Tiny** with a **multi-dataset fusion pipeline**, automatic leaf-mask generation, specialized loss functions, and extensive evaluation.

The goal is to move from *“where is the leaf?”* to *“how is the disease structured **within** the leaf?”*, aligning the model with actual plant biology.

---

## 📁 Dataset Description (DS)

The model uses a **multi-dataset fusion strategy**, integrating two complementary datasets to reflect real-world conditions.
Details below are derived from the dataset structure and methodology in the uploaded IEEE draft (pages 1–5) .

### **1. CVPPP 2017 Dataset**

💡 *Healthy leaf structure dataset*

* Contains annotated plant images for **leaf count**, **instance segmentation**, and **boundary masks**.
* Offers high-quality structural leaf annotations.
* Used primarily to teach the model accurate **leaf shapes**, **boundaries**, and **plant geometry**.

This dataset strengthens the *structural* understanding of the model.

---

### **2. PlantDoc Disease Dataset**

💡 *In-the-wild disease dataset with real variation*

* Provides **pixel-level masks for disease lesions** across diverse plant species.
* Includes multiple disease types, color variations, occlusions, noise, and natural backgrounds.
* Some masks are incomplete or coarse, so the pipeline uses **automatic leaf-mask synthesis** (LAB–HSV fusion + dilation) to generate reliable leaf priors (page 5) .

This dataset trains the model’s **disease-awareness** and ability to work on messy real-world images.

---

### **3. Final Combined Dataset**

* All images resized to **256×256**.
* Around **3000 training** + **750 validation** samples used.
* Combined into a unified triplet format:

  ```
  { image, leaf_mask, disease_mask }
  ```
* Automatic leaf-mask generator applied wherever ground-truth leaf masks are missing.

This fusion produces a dataset that respects the leaf–disease biological hierarchy and improves cross-domain generalization.

---

## 🏗️ Model Architecture Overview

Based on the architectural sections in the paper (pages 5–7) .

### 🔹 **ConvNeXt-Tiny Backbone**

* 4 hierarchical stages
* Depthwise convolutions
* LayerNorm + GELU
* DropPath + LayerScale
* Multi-resolution feature extraction

This backbone provides high representational strength with efficiency similar to modern CNN–Transformer hybrids.

### 🔹 **Dual-Headed Decoder**

1. **Leaf Segmentation Head**
2. **Disease Lesion Segmentation Head**

This allows the model to explicitly maintain the biological rule:
→ *Disease must occur within leaf boundaries.*

### 🔹 **Key Innovations**

* Automatic leaf mask generation (LAB–HSV vegetation cues + disease-aware dilation)
* Hierarchy Consistency Loss (punishes disease pixels outside leaves)
* Spatial Smoothness Loss for clean boundaries
* Weighted BCE for rare disease pixels
* Multi-resolution upsampling to recover **256×256** fidelity

---

## 🧪 Methodology Summary

Condensed from the workflow and methodology on pages 3–7 .

1. **Data Preparation**

   * Augmentations: flips, affine transforms, jitter, noise, blur
   * Auto leaf-mask generation for PlantDoc images
   * Multi-dataset fusion
   * PyTorch DataLoaders with hierarchical collation

2. **Model Training**

   * Combined hierarchical loss
   * AdamW optimizer
   * Cosine annealing LR scheduler
   * Mixed precision training
   * Checkpoints + validation monitoring

3. **Evaluation**

   * IoU, Dice, F1
   * Threshold tuning
   * Disease severity distribution
   * Confidence calibration
   * Visual overlays and heatmaps

4. **Testing**

   * Reload best checkpoint
   * Generate final leaf + disease masks
   * Validate hierarchical alignment visually and numerically

---

## 📊 Final Results

Metrics below come from your **res_metrics.txt file**  and summarize the disease-detection ability of two backbones: **ResNet** and **ConvNeXt Base**.

### ### **1. ResNet Disease Metrics (Threshold = 0.2)**

* Sensitivity: **0.9133**
* Specificity: **0.7759**
* Precision: **0.4420**
* NPV: **0.9788**
* Accuracy: **0.7983**
* F1: **0.5957**
* IoU: **0.4242**
* Dice: **0.5957**
* Fallout: **0.2241**

**Best threshold = 0.95:**

* F1: **0.6273**
* IoU: **0.4570**

---

### **2. ConvNeXt Base Disease Metrics (Threshold = 0.2)**

* Sensitivity: **0.8675**
* Specificity: **0.8955**
* Precision: **0.6173**
* NPV: **0.9720**
* Accuracy: **0.8909**
* F1: **0.7213**
* IoU: **0.5641**
* Dice: **0.7213**
* Fallout: **0.1045**

**Best threshold = 0.95:**

* F1: **0.7493**
* IoU: **0.5991**

---

## 🏆 Interpretation

* **ConvNeXt Base outperforms ResNet** in nearly all metrics except raw sensitivity.
* It produces much more stable training, better leaf segmentation IoU, and more reliable disease predictions.
* ResNet has higher recall but suffers from over-segmentation and instability.

Together, they validate the choice of **ConvNeXt-Tiny** in your hierarchical model.

---

## 🚀 Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/plant-hierarchy-segmentation

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py

# Run evaluation
python evaluate.py

# Run inference
python infer.py --image sample.jpg
```
