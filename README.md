# H. Pylori Contamination Detection (Iteration 9.2)

This project implements a **Dual-Stage Clinical Diagnostic Pipeline** for the automated detection of *H. pylori* contamination in IHC tissue samples.

## Project Structure

- `dataset.py`: High-performance data loader optimized for IHC patches.
- `model.py`: Backbone architecture using **ConvNeXt-Tiny** with a non-linear classification head (ReLU + Dropout).
- `meta_classifier.py`: **Clinical Meta-Layer** using a Random Forest (17-feature signature) optimized via Leave-One-Patient-Out (LOPO) cross-validation.
- `train.py`: Main engine fork-lifting:
    - **5-Fold Cross-Validation**: Rigorous validation strategy by Patient ID.
    - **OneCycleLR**: Specialized learning rate scheduling for rapid convergence.
    - **ImageNet Normalization**: Shifted from Macenko to standard normalization for improved backbone stability.
- `normalization.py`: Support for GPU-vectorized stain normalization.
- `generate_visuals.py`: Clinical reporting engine (ROC/PR curves, Grad-CAM heatmaps).

## Performance (Iteration 9.2 Breakthrough)

The pipeline has achieved **Clinical-Grade Reliability** by implementing "Spatial De-Noising" and Meta-Optimization:

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Patient Accuracy** | **92.41%** | ↑ Project Peak |
| **Clinical Precision** | **94.57%** | ✓ Minimized False Contaminations |
| **Sensitivity (Recall)** | **90.00%** | ✓ High Detection Rate |
| **Throughput (A40)** | **~728 images/sec** | Optimized for ROI scanning |

## Diagnostic Architecture: The 17-Feature Meta-Layer

To break the 92% barrier, this model replaces manual heuristic "gates" with a **Random Forest Meta-Classifier**. It analyzes the statistical distribution of all patches within a patient folder:

1.  **Probabilistic Density**: Analyzes the Mean, Max, and Standard Deviation of patch scores.
2.  **Count-Based Signal**: Tracks high-confidence "suspicious" patches at multiple thresholds (P ≥ 70%, 80%, 90%).
3.  **Statistical Moments**: Calculates Skewness and Kurtosis of the probability distribution to differentiate between "sparse bacteremia" and "stain artifacts."

## Hardware & Training Strategy
- **Backbone**: `convnext_tiny` (pre-trained on ImageNet-1K).
- **Scheduler**: `OneCycleLR` (Max LR: 5e-4).
- **Augmentation**: Geometric (Rotate, Flip) + Color Jittering (0.2) + Morphological (Blur).
- **Compute**: Optimized for **NVIDIA A40 (48GB)** using a batch size of 128 (accumulation steps: 2) at 448x448 resolution, achieving **5.69 iterations/second** (~728 images/sec).

## How to Get Started

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Full Pipeline (SLURM)**:
   The easiest way to replicate results is using the submission scripts:
   ```bash
   bash submit_all_folds.sh  # Runs 5-fold training in parallel
   ```

3. **Running the Meta-Analysis**:
   If the meta classifier did not run automatically after training all folds, it can be done manually:
   ```bash
   python3 meta_classifier.py
   ```

## Key Clinical Findings
- **Spatial Independence**: Removing sparse spatial metadata (X, Y coords) improved accuracy by 0.34% and precision by 1.00%, confirming that morphological signal density is the primary diagnostic driver.
- **Max_Prob Dominance**: The single most confident patch remains the strongest predictor (24.14% relative importance) in the ensemble logic.

## Future Research (Iteration 10)
- **Attention-MIL (Bag-Level MIL)**: Instead of feeding the model one patch at a time, you feed it a "bag" of 500 patches from the same patient. The CNN extracts features from all 500. Then, the AttentionGate looks at all 500 simultaneously, figures out which 5 actually contain bacteria, assigns them a mathematical "weight," and outputs a single Patient-Level Diagnosis directly.

    Why it's better: It completely removes the need for a Meta-Classifier and heuristic rules. The neural network itself learns how to aggregate the data. It learns, natively, that 495 patches are irrelevant background tissue, and optimizes its gradients solely based on the 5 suspicious patches.

- **Test-Time Augmentation**: Augment the patches 8 different ways during verification then average the results to minimize artifacts.

The Right Way (Feature-Level Averaging)

    Step 1: Take Patch 1. Create the 8 TTA versions (rotations/flips).

    Step 2: Pass those 8 images through the ConvNeXt backbone to get 8 feature vectors (shape: [8, 768]).

    Step 3: Calculate the mean of those 8 vectors (shape becomes [1, 768]). This is now your TTA-smoothed feature vector for Patch 1.

    Step 4: Repeat for all 500 patches.

    Step 5: Feed the final bag of 500 smoothed feature vectors (shape: [500, 768]) into the Attention Gate.
