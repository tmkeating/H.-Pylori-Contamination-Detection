# H. Pylori Tissue Classification: Project Overview

This project uses Deep Learning to detect the presence of *H. pylori* bacteria in cell tissue samples. Below is a plain-English guide to the model, followed by a technical deep-dive for Subject Matter Experts.

---

## 1. General Overview (for Non-Experts)

### **Objective**
The goal is to build an artificial intelligence "brain" that can look at high-resolution **Annotated** digital microscopic images (patches) and determine if they are **Contaminated** (positive) or **Negative** (clean).

### **The Strategy: "Method 1: Fully Pre-trained Transfer Learning"**
Instead of trying to teach a brain to see from scratch, we use **Transfer Learning**.
1. We take a famous model called **ResNet18**, which has already been "raised" by looking at 1 million common images.
2. Because it already knows how to see shapes, edges, and textures, we only have to "fine-tune" its final layer to specialize in medical pathology.
3. **Upscaling**: We upscale the tissue patches from their original small size to $448 \times 448$ pixels. This allows the pre-trained ResNet filters to "zoom in" on the bacterial structures more effectively than the standard ImageNet resolution would.

### **Quick Summary of the Process**
*   **Data Scale (Expanded Training)**: We have expanded the dataset to ~54,000 images. This includes the ~2,700 "Annotated" high-fidelity patches supplemented by ~50,000 negative patches from the `Cropped` folders of confirmed healthy patients.
*   **Data Quality**: We maintain a strict hierarchy of truth. Positive labels ONLY come from pathologist-verified annotations. Negative labels come from either verified annotations or from patients with a confirmed 100% negative diagnosis.
*   **The Learning Process**: The computer looks at an image, makes a guess, and compares it to the "Answer Key" verified by pathologists. 
*   **Evaluation (HoldOut split)**: Evaluation is performed on a 20% stratified split of the high-fidelity data, ensuring we have a balanced "Final Exam" that includes confirmed bacteria.

---

## 2. Subject Matter Expert (SME) Deep Dive

### **Architecture: ResNet18 (Residual Network)**
*   **Type**: Convolutional Neural Network (CNN).
*   **Unique Feature**: "Skip Connections" (Residuals).
*   **Resolution**: Optimized at $448 \times 448$ input size.

### **Optimization Strategy**
*   **Loss Function**: **Weighted Cross-Entropy Loss**. We assign a balanced weight to the "Contaminated" class to optimize for both high sensitivity and high specificity.
*   **Imbalance Handling**: 
    *   **Weighted Loss ($w=2.0$)**: Fine-tuned to maintain 100% Recall while pushing Precision above 95% by reducing false alarms from tissue artifacts.
    *   **WeightedRandomSampler**: Manages the extreme $1:50$ imbalance in the raw data to ensure every training batch is statistically balanced ($1:1$ ratio).
*   **Optimizer**: **Adam** with a refined Learning Rate of $5 \times 10^{-5}$ over **15 epochs** for stable convergence.

### **Data Pipeline & Labeling**
*   **Dataset Source**: Dual source ( `Annotated` + `Cropped` Negatives).
*   **Labeling Prioritization**:
    1.  **Level 1 (Annotated)**: Top priority. Filenames map to Window IDs in Excel.
    2.  **Level 2 (Healthy Supplementation)**: All patches from "NEGATIVA" patients are included to improve the model's ability to recognize various healthy tissue textures.
    3.  **Level 3 (Silence)**: Unannotated patches from positive patients are discarded to prevent label noise.
*   **Normalization**: Standard ImageNet $(\mu, \sigma)$.
*   **Augmentation**: Horizontal/Vertical flips, Random Rotations, and Color Jitter.

### **Infrastructure & Deployment**
*   **SLURM Integration**: Configured for high-memory clusters (tasks: 8 cores, 32GB RAM).
*   **Result Versioning**: Every training run automatically creates a versioned report (e.g., `results/06_101764_evaluation_report.csv`).
*   **Hardware Compatibility**: Optimized for CPU/GPU. Support for **Intel Extension for PyTorch (IPEX)** is integrated for optimized inference on CPU-only nodes.

---

## 3. Training Performance Estimations
*   **Current Goal**: Achieve >80% Recall on the HoldOut set while maintaining >60% Precision.
*   **Deployment & Visual Diagnostics**: 
    *   **Standard Metrics**: Results are saved as machine-readable CSVs, Confusion Matrices (PNG), and ROC curves (PNG).
    *   **Advanced Interpretability & Reporting**: 
        *   **Grad-CAM Heatmaps**: Automatically generated for positive samples to show the specific tissue areas driving the "Contaminated" prediction.
        *   **Precision-Recall (PR) Curves**: Critical for evaluating performance in highly imbalanced datasets.
        *   **Probability Histograms**: Visualizes the model's confidence distribution across both classes.
        *   **Learning Curves**: Used to monitor training stability and convergence across 10 epochs.