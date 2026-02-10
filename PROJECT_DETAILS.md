# H. Pylori Tissue Classification: Project Overview

This project uses Deep Learning to detect the presence of *H. pylori* bacteria in cell tissue samples. Below is a plain-English guide to the model, followed by a technical deep-dive for Subject Matter Experts.

---

## 1. General Overview (for Non-Experts)

### **Objective**
The goal is to build an artificial intelligence "brain" that can look at digital microscopic images of tissue samples and determine if they are **Contaminated** (positive) or **Negative** (clean).

### **The Strategy: "Method 1: Fully Pre-trained Transfer Learning"**
Instead of trying to teach a baby brain to see from scratch, we use **Transfer Learning**.
1. We take a famous model called **ResNet18**, which has already been "raised" by looking at 1 million common images (like cats, bicycles, and scenery). 
2. Because it already knows how to see shapes, edges, and textures, we only have to "fine-tune" its final layer to specialize in medical pathology.
3. This is much faster and requires much less data than starting from zero.

### **Quick Summary of the Process**
*   **Data Preparation**: We feed the computer images from two sets: a training set (the "Study Material") and a hold-out set (the "Final Exam").
*   **The Learning Process**: The computer looks at an image, makes a guess, and compares it to the "Answer Key" (the CSV labels). 
*   **Optimization**: If the guess is wrong, the computer uses complex math (Calculus) to slightly adjust its internal connections to do better next time.
*   **Evaluation**: Once the training is done, we test it on images it has never seen before to see if it truly learned or just memorized the answers.

---

## 2. Subject Matter Expert (SME) Deep Dive

### **Architecture: ResNet18 (Residual Network)**
*   **Type**: Convolutional Neural Network (CNN).
*   **Unique Feature**: "Skip Connections" (Residuals). These allow gradients to flow through very deep networks without "vanishing," which prevents the model from forgetting earlier layers during backpropagation.
*   **Modification**: The original `fc` (fully connected) head was replaced from a 1000-way ImageNet classifier to a binary logit output ($z \in \mathbb{R}^2$), which is then mapped via a Softmax function to $P(y=1|x)$.

### **Optimization Strategy**
*   **Loss Function**: **Weighted Cross-Entropy Loss**. We assign a significantly higher weight ($w=5.0$) to the "Contaminated" class. This prioritizes **Recall (Sensitivity)** over Precision, ensuring that the penalty for a False Negative (missed infection) is five times higher than a False Positive.
*   **Sampling Strategy**: **WeightedRandomSampler**. To combat data imbalance, we use an oversampling technique where positive samples are sampled more frequently. This ensures that every training batch is statistically balanced ($1:1$ ratio), providing the model with more opportunities to learn the features of contaminated tissue.
*   **Optimizer**: **Adam (Adaptive Moment Estimation)**. Unlike standard Gradient Descent, Adam maintains separate learning rates for each parameter, adapting them based on the first and second moments of the gradients. This handles noisy microscopic data much more effectively.
*   **Learning Rate**: Set to $1 \times 10^{-4}$ to ensure stable convergence during fine-tuning.

### **Data Management & Normalization**
*   **Preprocessing**: All images are resized to $224 \times 224$ pixels.
*   **Normalization**: Pixel values are standardized using the ImageNet mean $(\mu = [0.485, 0.456, 0.406])$ and standard deviation $(\sigma = [0.229, 0.224, 0.225])$. This is critical because the pre-trained weights were learned on data with this specific distribution.
*   **Augmentation**: During training, we use **Random Horizontal and Vertical Flips**. Since medical pathology slides don't have a fixed "up" or "down," this prevents the model from learning orientation-based biases.

### **Evaluation Metrics**
*   **ROC Curve (Receiver Operating Characteristic)**: Tracks the trade-off between Sensitivity (True Positive Rate) and Specificity (1 - False Positive Rate) across all probability thresholds.
*   **AUC (Area Under the Curve)**: Provides a single scalar metric for classifier quality independent of the classification threshold.
*   **HoldOut Set**: A strictly separated group of patients (not just patches) to ensure the model generalizes across patients rather than just memorizing local slide textures.

---

## 3. Training Performance Estimations
*   **Hardware**: Intel Core Ultra 7 258V (Lunar Lake).
*   **Inference Latency**: Sub-10ms per image using iGPU/NPU acceleration via IPEX.
*   **Quantization Potential**: This model can be further optimized via INT8 quantization for real-time deployment on mobile pathology scanners.
