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
*   **Data Quality (Annotated Set)**: We use a high-fidelity dataset of ~2,700 patches where every image name maps directly to an entry in the pathologist's annotation Excel. This ensures every "1" (Contaminated) is a verified bacterial colony and every "0" (Negative) is confirmed healthy tissue.
*   **Data Filtration**: We only train on patches that have specific coordinate-based annotations (**Verified Positives/Negatives**) or patches from patients confirmed to be entirely **Negative**. We skip unannotated regions from positive patients to prevent "label poisoning."
*   **The Learning Process**: The computer looks at an image, makes a guess, and compares it to the "Answer Key" verified by pathologists. 
*   **Evaluation (HoldOut split)**: Because the external HoldOut data lacks verified positive labels, we evaluate performance using a strictly separated 20% split of the **Annotated** dataset. This ensures our "Final Exam" actually tests the model's ability to find bacteria.

---

## 2. Subject Matter Expert (SME) Deep Dive

### **Architecture: ResNet18 (Residual Network)**
*   **Type**: Convolutional Neural Network (CNN).
*   **Unique Feature**: "Skip Connections" (Residuals).
*   **Resolution**: Optimized at $448 \times 448$ input size. This increases the receptive field detail for detecting fine bacterial filaments while maintaining compatibility with ResNet's global average pooling.
*   **Modification**: The original `fc` (fully connected) head was replaced with a binary output.

### **Optimization Strategy**
*   **Loss Function**: **Weighted Cross-Entropy Loss**. We assign a significantly higher weight ($w=10.0$) to the "Contaminated" class. This prioritizes **Recall (Sensitivity)**, ensuring that the penalty for a False Negative (missed infection) is ten times higher than a False Positive.
*   **Imbalance Handling**: 
    *   **Weighted Loss ($1:10$)**: Penalizes misses heavily.
    *   **WeightedRandomSampler**: Ensures every training batch is statistically balanced ($1:1$ ratio).
    *   **Early Stopping Metric**: The model is saved based on **Minimum Validation Loss** (which considers the 10x penalty) rather than raw Accuracy.
*   **Optimizer**: **Adam (Adaptive Moment Estimation)** with a Learning Rate of $1 \times 10^{-4}$.

### **Data Pipeline & Labeling**
*   **Dataset Source**: `Annotated` folder (~2,700 verified images).
*   **Labeling Prioritization**:
    1.  **Level 1 (Annotated)**: Window IDs (e.g., `00902.png` normalized to `902`) in the patch Excel are used directly.
    2.  **Level 2 (Confirmed Negative)**: All patches from patients with "NEGATIVA" diagnosis are used as 0.
    3.  **Level 3 (Ambiguous)**: Unannotated patches from positive patients are **discarded** to ensure zero label noise.
*   **Normalization**: Standard ImageNet mean/std $(\mu, \sigma)$ applied via `torchvision.transforms`.
*   **Augmentation**: Horizontal/Vertical flips, Random Rotations, and Color Jitter to handle stain variability.

### **Infrastructure & Deployment**
*   **SLURM Integration**: Configured for high-memory clusters (tasks: 8 cores, 32GB RAM).
*   **Result Versioning**: Every training run automatically creates a versioned report (e.g., `results/06_101764_evaluation_report.csv`).
*   **Hardware Compatibility**: Optimized for CPU/GPU. Support for **Intel Extension for PyTorch (IPEX)** is integrated for optimized inference on CPU-only nodes.

---

## 3. Training Performance Estimations
*   **Current Goal**: Achieve >80% Recall on the HoldOut set while maintaining >60% Precision.
*   **Deployment**: Results are saved as machine-readable CSVs, confusion matrices (PNG), and ROC curves (PNG) for rapid clinical review.