# Brain Tumor Detection

Benchmarking CNN architectures for multi-class brain tumor classification from MRI scans. Presented at **IC3SE 2025**.

---

## Problem Statement

Brain tumors are among the most critical neurological conditions, and early detection significantly improves patient outcomes. Manual analysis of MRI scans is time-consuming and prone to human error. This project benchmarks multiple deep learning architectures to classify brain MRI images into four categories — **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor** — and identifies the best-performing model for clinical decision support.

## Pipeline

```
MRI Images (224x224x3)
        |
   Preprocessing
   (Rescale 1/255, Brightness Augmentation)
        |
   Train / Val / Test Split
   (4570 / 1142 / 1311 images)
        |
   ┌────────────┬────────────┬────────────┬────────────┬────────────┐
   |  Xception  | EfficientNet|  ResNet-50 |  Custom CNN|  Dilated   |
   |            |     B4      |            | (Residual) |    CNN     |
   └────────────┴────────────┴────────────┴────────────┴────────────┘
        |
   Evaluation
   (Accuracy, Precision, Recall, F1, AUC-ROC)
        |
   Confusion Matrix + Classification Report
```

## Dataset

**Source:** [Kaggle — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|-------|--------|------------|----------|-----------|-------|
| Train | 1,053 | 1,083 | 1,297 | 1,137 | 4,570 |
| Val | 268 | 256 | 298 | 320 | 1,142 |
| Test | 300 | 306 | 405 | 300 | 1,311 |

**4 classes** across 7,023 MRI images total.

## Models

### Transfer Learning Models

All pretrained on ImageNet, fine-tuned end-to-end with a custom classification head:

```
Base Model (ImageNet) → Flatten → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.25) → Dense(4, Softmax)
```

- **Xception** — Depthwise separable convolutions
- **EfficientNetB4** — Compound scaling (depth + width + resolution)
- **ResNet-50** — Residual skip connections

### Custom CNN (Residual)

Built from scratch with dilated convolutions and residual blocks:

```
Conv2D(32) → [ResBlock(64) → ResBlock(128) → ResBlock(256) → ResBlock(512)] → GAP → Dense(256) → Dense(4)
```

- Dilated convolutions (rates 2, 3) for larger receptive fields
- L2 regularization + Dropout(0.6)

### Dilated CNN

Sequential architecture with progressively increasing filters and dilated convolutions:

```
Conv2D(64, d=3) ×2 → Conv2D(128, d=2) ×2 → Conv2D(256, d=1) ×2 → Dense(512) → Dense(4)
```

## Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| **Xception** | **97.7%** | **98%** | **98%** | **98%** | **99.6%** |
| EfficientNetB4 | ~98.7%* | — | — | — | 99.9% |
| ResNet-50 | ~97.9%* | — | — | — | 99.5% |
| Custom CNN (Residual) | 97.1% | 97% | 97% | 97% | 99.5% |
| Dilated CNN | 94.9% | 95% | 95% | 95% | 99.3% |

*\*Partial output captured during evaluation*

### Per-Class Breakdown (Xception — Best Full Report)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Glioma | 1.00 | 0.93 | 0.96 | 300 |
| Meningioma | 0.93 | 0.98 | 0.95 | 306 |
| No Tumor | 1.00 | 1.00 | 1.00 | 405 |
| Pituitary | 0.97 | 1.00 | 0.99 | 300 |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Input Size | 224 x 224 x 3 |
| Batch Size | 64 |
| Optimizer | Adamax |
| Learning Rate | 0.001 (transfer) / 0.0005 (custom) |
| Loss | Categorical Crossentropy |
| Epochs | 50 (with early stopping) |
| Early Stopping | patience=10, monitor=val_loss |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Augmentation | Brightness range (0.8–1.2) |

## Tech Stack

- Python, TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn (classification report, confusion matrix)

## Publication

Presented at **IC3SE 2025** (International Conference on Communication, Computing, and Smart Systems Engineering).

## Getting Started

```bash
git clone https://github.com/sidd707/Brain-tumor-Detection.git
cd Brain-tumor-Detection
```

Open `brain-tumor-mri-dataset.ipynb` in Jupyter or [Kaggle](https://www.kaggle.com/) to reproduce results.

## License

This project is for academic and research purposes.
