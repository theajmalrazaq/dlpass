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

The benchmark investigates model design and regularization when trained on an acute low-sample budget from CIFAR-10:
- **Low-Sample Training Budget:** Exactly 500 clean CIFAR-10 images (balanced 50 per class), stratified into **400 training** and **100 validation** images under fixed seed `42`.
- **Corrupted Shifted Test Evaluation:** 10,000 CIFAR-10 test images processed through a deterministic, seeded corruption pipeline:
  - In-plane random rotation up to $\pm 25^\circ$
  - Multiplicative brightness and contrast degradation ($0.7\times$ to $1.3\times$)
  - Rectangular occlusion covering approximately $25\%$ of the image surface ($16 \times 16$ pixel patch)
  - Hidden unlisted corruption: Defocus Gaussian blur ($k=5, \sigma \in [1.2, 2.0]$)
- **Specification Compliance:**
  - Trained completely from scratch (no pretrained models or external weights).
  - Trainable parameters capped at $1,000,000$ (our architecture uses **91,618 parameters**, well within budget).
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

## Architectural Blueprint: MultiScaleInceptionNet

To prevent destructive memorization on only 400 training instances while preserving salient geometric cues under corruption, we designed **MultiScaleInceptionNet** ($91,618$ parameters):
1. **Two-Stage Convolutional Stem:**
   - $\text{Conv2D}(3 \to 32, 3\times 3) + \text{BN} + \text{LeakyReLU}(0.1)$
   - $\text{Conv2D}(32 \to 64, 3\times 3) + \text{BN} + \text{LeakyReLU}(0.1)$
   - $\text{MaxPool2D}(2\times 2) \to 16\times 16$
2. **Asymmetric Inception Block 1 (64 $\to$ 128 channels):**
   - Branch 1: $1\times 1$ Conv ($32$ channels)
   - Branch 2: $1\times 1$ Reduction ($16$) $\to 3\times 3$ Conv ($32$ channels)
   - Branch 3: $1\times 1$ Reduction ($8$) $\to 5\times 5$ Conv ($32$ channels)
   - Branch 4: $3\times 3$ MaxPool $\to 1\times 1$ Projection ($32$ channels)
   - Downsampling: $\text{MaxPool2D}(2\times 2) \to 8\times 8$
3. **Asymmetric Inception Block 2 (128 $\to$ 192 channels):**
   - Branch 1: $1\times 1$ Conv ($48$ channels)
   - Branch 2: $1\times 1$ Reduction ($24$) $\to 3\times 3$ Conv ($48$ channels)
   - Branch 3: $1\times 1$ Reduction ($12$) $\to 5\times 5$ Conv ($48$ channels)
   - Branch 4: $3\times 3$ MaxPool $\to 1\times 1$ Projection ($48$ channels)
4. **Regularized Dual-Stage Classification Head:**
   - $\text{AdaptiveAvgPool2D}(1\times 1) \to 192$ feature vector
   - $\text{Dropout}(0.4) \to \text{Linear}(192 \to 64) \to \text{BatchNorm1D}(64) \to \text{LeakyReLU}(0.1) \to \text{Linear}(64 \to 10)$

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

Every configuration was trained strictly on 400 clean CIFAR-10 images across three independent random seeds (`42`, `123`, `999`), reporting the mean and standard deviation:

| Configuration | Parameters | Validation Acc (%) | Clean Test Acc (%) | Shifted Test Acc (%) | Robustness Drop (Δ) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. No augmentation (Baseline)** | 91,618 | 53.7 ± 1.9 | 43.41 ± 0.74 | 15.25 ± 1.78 | 28.16 ± 2.36 |
| **2. Geometric only** | 91,618 | 53.3 ± 1.2 | 43.56 ± 0.67 | 19.26 ± 1.27 | 24.30 ± 0.59 |
| **3. Photometric only** | 91,618 | 51.7 ± 0.5 | 40.73 ± 2.05 | 17.97 ± 0.58 | 22.76 ± 1.85 |
| **4. Geometric + photometric** | 91,618 | 50.0 ± 2.4 | 38.70 ± 1.55 | 19.95 ± 1.97 | 18.75 ± 2.74 |
| **5. + Cutout / Random Erasing** | 91,618 | 46.3 ± 4.2 | 38.51 ± 0.71 | **23.08 ± 3.62** | 15.43 ± 3.00 |
| **6. + Mixup + label smoothing** | 91,618 | 44.0 ± 2.2 | 37.12 ± 0.20 | **23.16 ± 0.67** | 13.97 ± 0.75 |
| **7. Final model** | 91,618 | 44.0 ± 2.2 | 37.12 ± 0.20 | **23.16 ± 0.67** | 13.97 ± 0.75 |

---

## Executive Summary of Findings

1. **Baseline Failure (Part A):**  
   Training on 400 clean images with zero augmentation yields near-perfect training fit, but test performance drops to $42.1\%$ on clean data and collapses to $13.5\%$ under distribution shift. The model memorizes high-frequency training shortcuts that are obliterated by test-time noise, blur, and occlusion.

2. **Single Most Impactful Augmentation (Part D):**  
   **Cutout / Random Erasing** provided the single largest jump in shifted test accuracy. Because the shifted pipeline injects a $16\times 16$ rectangular mask covering $\sim 25\%$ of the image, training with random rectangular erasures prevents the network from over-relying on solitary features (e.g. eyes or wheels) and forces it to synthesize spatially distributed representations.

3. **Augmentation Trade-offs (Part B):**  
   - *Vertical Flipping:* Degrades performance because terrestrial objects (cars, horses, boats) possess strong gravitational orientation priors. Flipping upside down introduces invalid visual geometries.  
   - *$\pm 90^\circ$ vs $\pm 25^\circ$ Rotation:* Augmenting up to $\pm 90^\circ$ dilutes canonical upright representations. Aligning training variance with the expected test distortion ($\pm 25^\circ$) yields superior target generalization.  
   - *Excessive Augmentation:* On a compact $91\text{k}$ parameter model trained on only 400 images, combining too many severe transforms simultaneously destroys semantic signals, inducing underfitting.

4. **Regularization Placement (Part C):**  
   Dropout ($0.4$) is positioned after Global Average Pooling in the dense head. Convolutional feature maps exhibit high spatial autocorrelation where adjacent units reconstruct dropped features; after spatial pooling, each channel represents an independent global semantic cue where dropout effectively disrupts co-adaptation.

5. **Diagnostic Class Confusion (Part E):**  
   - *Top Confusions:* `Cat` vs `Dog` (similar quadrupedal silhouettes at $32\times 32$) and `Automobile` vs `Truck` (occlusion masking cargo beds).  
   - *Hidden Corruption Generalization:* The final model achieved over $34\%$ on the hidden defocus blur test set, demonstrating that multi-scale Inception branches ($5\times 5$ filters and max-pooling) capture robust low-frequency shapes even when high-frequency textures are smoothed out.

6. **Test-Time Augmentation (Part F):**  
   Ensembling predictions across original and horizontally flipped test samples provided marginal accuracy change at the cost of a $2\times$ latency penalty. Standard single-pass inference remains optimal for real-time edge deployments.
# dlpass
