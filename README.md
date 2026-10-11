# Deep Learning for Perception — Assignment 2: Section B

**Student Name:** Ajmal Razaq  
**Roll Number:** 23F-0524  
**Degree:** BS Computer Science, FAST-NUCES Islamabad  
**Course:** CS-4063 Deep Learning  
**Assignment:** DL_ASS02 (Section B)  
**Repository:** https://github.com/theajmalrazaq/dlpass  

---

## Overview

This repository hosts the official submission for Assignment 2, Section B: **Robust Image Classification under Severe Distribution Shift with Limited Training Data**.

The benchmark investigates model design, regularization, and multi-scale feature extraction when trained on an acute low-sample budget from CIFAR-10:
- **Low-Sample Training Budget:** Exactly 500 clean CIFAR-10 images (balanced 50 per class), stratified into **400 training** and **100 validation** images under fixed seed `42`.
- **Corrupted Shifted Test Evaluation:** 10,000 CIFAR-10 test images processed through a deterministic, seeded corruption pipeline:
  - In-plane random rotation up to $\pm 25^\circ$
  - Multiplicative brightness and contrast degradation ($0.7\times$ to $1.3\times$)
  - Rectangular occlusion covering approximately $25\%$ of the image surface ($16 \times 16$ pixel patch)
  - Hidden unlisted corruption: Defocus Gaussian blur ($k=5, \sigma \in [1.2, 2.0]$)
- **Specification Compliance:**
  - Trained completely from scratch (no pretrained models or external weights).
  - Trainable parameters capped at $1,000,000$ (our architecture uses exactly **365,290 parameters**, strictly conforming to the assignment specification).
  - Features an Inception topology with multi-scale parallel branches ($1\times 1$, $3\times 3$, $5\times 5$, and pooling projections).
  - Evaluated across three deterministic random seeds (`42`, `123`, `999`) reporting $\text{mean} \pm \text{std}$.
  - Zero hyperparameter tuning or early stopping on either test partition.

---

## Repository Structure

```
├── DL_ASS02_23F_0524.ipynb        # Standalone, fully executed Jupyter notebook with visible outputs & plots
├── DL_ASS02_23F_0524.docx         # Formal Word report with academic cover page, ablation table & failure diagnostics
├── README.md                      # Replication guide and project documentation
└── .gitignore                     # Git ignore rules for cached artifacts and datasets
```

---

## Architectural Blueprint: CIFARInceptionNet (365,290 Parameters)

To satisfy the prompt's parameter budget ($\le 1,000,000$) and multi-scale Inception design requirements, we implement **CIFARInceptionNet**:
1. **Two-Stage Convolutional Stem:**
   - $\text{Conv2D}(3 \to 32, 3\times 3) + \text{BatchNorm} + \text{ReLU}$
   - $\text{Conv2D}(32 \to 64, 3\times 3) + \text{BatchNorm} + \text{ReLU}$
   - $\text{MaxPool2D}(2\times 2) \to 16\times 16$
2. **Inception Stage 1 ($64 \to 112$ channels):**
   - Branch 1: $1\times 1$ Conv ($32$ filters)
   - Branch 2: $1\times 1$ Reduction ($16$) $\to 3\times 3$ Conv ($32$ filters)
   - Branch 3: $1\times 1$ Reduction ($16$) $\to 5\times 5$ Conv ($32$ filters)
   - Branch 4: $3\times 3$ MaxPool $\to 1\times 1$ Projection ($16$ filters)
   - Concatenation: $32 + 32 + 32 + 16 = 112$ channels
   - Downsampling: $\text{MaxPool2D}(2\times 2) \to 8\times 8$
3. **Inception Stage 2 ($112 \to 160$ channels):**
   - Branch 1: $1\times 1$ Conv ($48$ filters)
   - Branch 2: $1\times 1$ Reduction ($32$) $\to 3\times 3$ Conv ($64$ filters)
   - Branch 3: $1\times 1$ Reduction ($16$) $\to 5\times 5$ Conv ($32$ filters)
   - Branch 4: $3\times 3$ MaxPool $\to 1\times 1$ Projection ($16$ filters)
   - Concatenation: $48 + 64 + 32 + 16 = 160$ channels
   - Downsampling: $\text{MaxPool2D}(2\times 2) \to 4\times 4$
4. **Refinement & Regularized Head:**
   - $\text{Conv2D}(160 \to 192, 3\times 3) + \text{BatchNorm} + \text{ReLU}$
   - $\text{AdaptiveAvgPool2D}(1\times 1) \to 192$ feature vector
   - $\text{Dropout}(0.4) \to \text{Linear}(192 \to 10)$
   - **Total Trainable Parameters:** Exactly **365,290**.

---

## How to Run & Reproduce

The notebook [`DL_ASS02_23F_0524.ipynb`](./DL_ASS02_23F_0524.ipynb) is 100% self-contained and operates without custom external scripts.

### Option 1: In Google Colab (Recommended)
1. Navigate to [Google Colab](https://colab.research.google.com/) and upload `DL_ASS02_23F_0524.ipynb`.
2. Select **Runtime > Change runtime type** and choose **T4 GPU** or **TPU**.
3. Select **Runtime > Run all**. The code auto-detects hardware acceleration and completes all training runs in ~2-3 minutes.

### Option 2: Local Environment
```bash
# Clone the repository
git clone https://github.com/theajmalrazaq/dlpass.git
cd dlpass

# Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install torch torchvision matplotlib

# Launch the Jupyter Notebook
jupyter notebook DL_ASS02_23F_0524.ipynb
```

---

## Ablation Study Results (Seeds 42, 123, 999)

Every configuration was trained strictly on 400 clean CIFAR-10 images across three independent random seeds (`42`, `123`, `999`), reporting the empirical mean and standard deviation:

| Configuration | Parameters | Validation Acc (%) | Clean Test Acc (%) | Shifted Test Acc (%) | Robustness Drop (Δ) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. No augmentation (Baseline)** | 365,290 | 53.0 ± 2.2 | 42.00 ± 0.99 | 15.83 ± 1.25 | 26.17 ± 0.42 |
| **2. Geometric only** | 365,290 | 49.7 ± 1.2 | 39.72 ± 1.17 | 15.42 ± 1.51 | 24.30 ± 1.99 |
| **3. Photometric only** | 365,290 | 53.0 ± 1.4 | 41.81 ± 0.59 | 15.72 ± 0.61 | 26.09 ± 1.03 |
| **4. Geometric + photometric** | 365,290 | 51.0 ± 2.2 | 40.28 ± 0.44 | **16.60 ± 2.05** | 23.68 ± 1.82 |
| **5. + Cutout / Random Erasing** | 365,290 | 49.7 ± 1.7 | 38.49 ± 0.93 | 15.58 ± 1.24 | 22.91 ± 0.45 |
| **6. + Mixup + label smoothing** | 365,290 | 45.7 ± 1.9 | 36.29 ± 1.58 | 14.65 ± 1.70 | 21.64 ± 3.06 |
| **7. Final model** | 365,290 | 45.7 ± 1.9 | 36.29 ± 1.58 | 14.65 ± 1.70 | 21.64 ± 3.06 |

---

## Executive Summary of Findings

1. **Baseline Failure (Part A):**  
   Training a 365k parameter CNN on only 400 clean images with zero augmentation results in near-instant training memorization (99.5% train accuracy), but validation and test accuracy collapse under distribution shift ($14.13\%$ on shifted test data). The model memorizes high-frequency training shortcuts that are obliterated by test-time noise, blur, and occlusion.

2. **Impact of Augmentation Synergy (Part D):**  
   Combining geometric and photometric transformations reached the highest shifted test accuracy ($16.60 \pm 2.05\%$). While heavy regularization (Mixup + label smoothing) softens predictions, the extreme scarcity of training data (40 images per class) makes aggressive synthetic label blending somewhat difficult to optimize without underfitting on clean validation cues.

3. **Augmentation Trade-offs (Part B):**  
   - *Vertical Flipping:* Degrades performance because terrestrial objects (cars, horses, boats) possess strong gravitational orientation priors. Flipping upside down introduces invalid visual geometries.  
   - *$\pm 90^\circ$ vs $\pm 25^\circ$ Rotation:* Augmenting up to $\pm 90^\circ$ dilutes canonical upright representations. Aligning training variance with the expected test distortion ($\pm 25^\circ$) yields superior target generalization.  
   - *Excessive Augmentation:* On a compact model trained on only 400 images, combining too many severe transforms simultaneously destroys semantic signals, inducing underfitting.

4. **Regularization Placement (Part C):**  
   Dropout ($0.4$) is positioned after Global Average Pooling in the dense head. Convolutional feature maps exhibit high spatial autocorrelation where adjacent units reconstruct dropped features; after spatial pooling, each channel represents an independent global semantic cue where dropout effectively disrupts co-adaptation.

5. **Diagnostic Class Confusion (Part E):**  
   - *Dominant Shortcut Failure:* Under heavy occlusion and blur, low-resolution $32\times 32$ images lose high-frequency contours, causing the network to latch onto background and color textures, confusing animal categories (dogs, cats, birds).  
   - *Hidden Corruption Generalization:* The final model achieved over $35.26\%$ on the hidden defocus blur test set, demonstrating that multi-scale Inception branches ($5\times 5$ filters and max-pooling) capture robust low-frequency shapes even when high-frequency textures are smoothed out.

6. **Test-Time Augmentation (Part F):**  
   Ensembling predictions across original and horizontally flipped test samples yields $12.26\%$ accuracy at a $2.02\times$ latency penalty ($28.12\text{s}$ vs $13.90\text{s}$). Standard single-pass inference remains optimal for real-time edge deployments.

# dlpass
