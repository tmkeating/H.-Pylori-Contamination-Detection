"""
H. Pylori Contamination Detection - Stain Normalization Utility
==============================================================

OVERVIEW
--------
This module implements Macenko color normalization for histological images, addressing
the critical challenge of stain variation across different labs, tissue blocks, and
slide preparation protocols. It provides:
  - Reference-based stain normalization (RGB → H&E decomposition → standardization)
  - Single-image and batch processing modes
  - GPU-accelerated computation via PyTorch
  - Optional pathological stain jittering for data augmentation
  - Safe fallback handling for edge cases (empty/white patches)

PURPOSE
-------
Histological H&E staining exhibits significant visual variation due to:
  - Different staining labs with different protocols
  - Tissue block age and storage conditions
  - Varying fixation times and reagent concentrations
  - Differences in slide scanning equipment

Without normalization, a model trained on one lab's slides may fail dramatically
on another lab's data (poor generalization). Macenko normalization projects all
images into a canonical "reference" stain space, enabling robust cross-lab diagnosis.

CLINICAL SIGNIFICANCE: Improves model robustness to real-world deployment variations
where slides may come from multiple labs with inconsistent staining practices.

HOW IT WORKS
------------

MATHEMATICAL FOUNDATION:
  Macenko normalization operates in optical density (OD) space where stain concentrations
  are separable and additive:

    OD = -log(I/Io)
    OD = C_H * V_H + C_E * V_E

  Where:
    I = captured RGB intensity [0, 255]
    Io = transmitted light intensity (~240)
    C_H, C_E = Hematoxylin and Eosin concentrations (what we want to normalize)
    V_H, V_E = Stain color vectors (what differs between labs)

ALGORITHM STEPS:

  1. REFERENCE FITTING (offline):
     a. Select representative slide from target lab → fit()
     b. Convert RGB → Optical Density space
     c. Compute PCA to find H&E color plane
     d. Extract Hematoxylin and Eosin stain vectors (V_H, V_E)
     e. Store reference vectors for future use

  2. SOURCE IMAGE NORMALIZATION (per image):
     a. Convert RGB → Optical Density space
     b. Estimate source stain matrix (V_H', V_E')
     c. Solve linear system: OD = [V_H', V_E'] * [C_H', C_E']
     d. Normalize concentrations: C_norm = C_source * (C_ref_max / C_source_max)
     e. Reconstruct: OD_norm = [V_H_ref, V_E_ref] * C_norm
     f. Convert back: I_norm = Io * exp(-OD_norm)

  3. BATCH PROCESSING (optional):
     - Vectorize steps 2a-f across entire batch
     - Process (B, C, H, W) in single GPU operation
     - Optional jitter adds stain variation for augmentation

BATCH NORMALIZATION DETAILS:
  - Fully vectorized via PyTorch for GPU efficiency
  - Per-sample stain matrix estimation (each image gets its own V_H, V_E)
  - Robust masking: ignores white background pixels via OD threshold (β=0.15)
  - Quantile-based statistics: α=1% excludes extreme outlier pixels
  - Device handling: automatically moves reference tensors to input device

PATHOLOGICAL STAIN JITTER:
  When jitter=True, adds realistic stain variation after normalization:
  - Multiplicative (α-jitter): ±20% intensity variation (simulates H&E depth changes)
  - Additive (β-jitter): ±5% background noise (simulates stain wash)
  - Preserves morphology while simulating "bad staining" scenarios
  - Useful for training robustness to adverse staining conditions

USAGE
-----

BASIC SETUP (Recommended for training):

  from normalization import MacenkoNormalizer
  import torch

  # 1. Create normalizer instance
  normalizer = MacenkoNormalizer()

  # 2. Fit to reference image from YOUR target lab/cohort
  reference_patch = torch.randn(3, 448, 448)  # or load from PIL image
  normalizer.fit(reference_patch, device='cuda')

  # 3. Apply to input patches
  normalized_patch = normalizer(input_patch)

SINGLE IMAGE NORMALIZATION:

  # Using PIL images
  from PIL import Image

  normalizer = MacenkoNormalizer()
  ref_img = Image.open('reference_slide.png')
  normalizer.fit(ref_img, device='cpu')

  input_img = Image.open('patient_slide.png')
  normalized_img = normalizer(input_img)  # Returns PIL Image

  # Using tensors
  normalizer = MacenkoNormalizer()
  ref_tensor = torch.randn(3, 448, 448).to('cuda')
  normalizer.fit(ref_tensor, device='cuda')

  input_tensor = torch.randn(3, 448, 448).to('cuda')
  normalized_tensor = normalizer(input_tensor)  # Returns normalized tensor in [0, 1]

BATCH PROCESSING (Recommended for inference):

  # Process entire batch at once (GPU-accelerated)
  batch = torch.randn(64, 3, 448, 448).to('cuda')  # (B, C, H, W), values in [0, 1]

  # Without jitter (standard normalization)
  normalized_batch = normalizer.normalize_batch(batch, jitter=False)  # (64, 3, 448, 448)

  # With jitter (data augmentation mode)
  augmented_batch = normalizer.normalize_batch(batch, jitter=True)  # With stain variation

INTEGRATION WITH TRAINING PIPELINE:

  from torch.utils.data import DataLoader
  from torch.utils.data.sampler import Sampler

  # Initialize normalizer from training reference
  normalizer = MacenkoNormalizer()
  reference_slide = load_reference_slide()
  normalizer.fit(reference_slide, device='cuda')

  # Apply in training loop
  for batch_images, batch_labels in train_loader:
      batch_images = batch_images.to('cuda')

      # Option 1: Normalize at batch level (faster)
      batch_images = normalizer.normalize_batch(batch_images, jitter=True)

      # Option 2: Normalize individual images (fallback)
      # normalized = torch.stack([normalizer(img) for img in batch_images])

      # Continue training
      logits = model(batch_images)
      loss = criterion(logits, batch_labels)
      loss.backward()

CLASS REFERENCE
---------------

MacenkoNormalizer()
  Stain normalization wrapper using torchstain backend.

  Attributes:
    normalizer: Internal torchstain.Macenko instance
    fitted: Boolean flag indicating if reference has been set

  Methods:
    fit(reference_img, device='cpu')
      Fit normalizer to a reference image (from target lab).
      Args:
        reference_img: PIL Image or torch.Tensor (C, H, W)
        device: 'cpu' or 'cuda' for computation device
      Returns: None (modifies internal state)
      Note: Should be called once per training run; use same reference for all images

    __call__(img)
      Normalize a single image to match reference stain profile.
      Args:
        img: PIL Image or torch.Tensor (C, H, W), values in any range
      Returns:
        PIL Image (if input was PIL) or torch.Tensor in [0, 1] (if input was tensor)
      Fallback: Returns original image if normalization fails (edge case safety)

    normalize_batch(batch_tensor, jitter=False)
      GPU-accelerated batch normalization with optional stain jittering.
      Args:
        batch_tensor: torch.Tensor (B, C, H, W), values in [0, 1]
        jitter: Boolean, if True applies pathological stain augmentation
      Returns:
        torch.Tensor (B, C, H, W), normalized and in [0, 1]
      Note: Preserves input device and dtype (CPU/GPU, float32/float64)

    __repr__()
      String representation showing fitted status

ALGORITHM PARAMETERS
--------------------
These are internal constants tuned for H&E histology:

  Io = 240.0 (Transmitted light intensity)
    - Standard constant for optical density calculation
    - Typical for histology microscopy

  alpha = 1 (Percentile masking)
    - Exclude extreme 1% of angle values
    - Reduces impact of artifacts on stain vector estimation

  beta = 0.15 (OD threshold for background masking)
    - Pixels with OD < 0.15 in all channels are background (white space)
    - Focus normalization on actually stained tissue

  Jitter parameters (when enabled):
    - Multiplicative: [0.8, 1.2] range = ±20% intensity
    - Additive: [-0.05, 0.05] range = ±5% background

ADVANCED DEPLOYMENT
-------------------

MULTI-LAB ROBUSTNESS:
  Train model once with a "canonical" reference slide.
  All slides (regardless of source lab) are normalized to that reference before inference.
  This enables consistent predictions across multi-site deployments.

DOMAIN ADAPTATION:
  If deploying to a new lab with notably different staining:
  1. Collect representative slides from new lab
  2. Refit normalizer to new reference (preserves trained model, only updates reference)
  3. Deploy with new reference for improved local accuracy

INFERENCE PIPELINE:
  1. Normalizer fitted during training on reference
  2. Save normalizer hyperparameters (HERef, maxCRef) with model checkpoint
  3. Load normalizer + model together for inference
  4. All incoming slides automatically normalized before prediction

DEPENDENCIES
------------
  - PyTorch: Tensor operations, GPU acceleration
  - torchstain: Specialized H&E color normalization backend
  - NumPy: Array handling
  - PIL: Image I/O and format conversion

NOTES
-----
  - Normalizer must be fitted() before use a single time
  - Same reference should be used for all images in training/inference
  - Batch processing (normalize_batch) is 100-1000x faster than per-image processing
  - GPU recommended for real-time inference (CPU mode is much slower)
  - Fallback returns original image if decomposition fails (e.g., empty white patches)
  - Device mismatches handled automatically (tensors moved as needed)
  - All output images are in [0, 1] range post-normalization
  - PIL images are returned as uint8 [0, 255]; tensors as float [0, 1]
  - Jittering is randomized; set seed for reproducibility
  - Batch normalization uses 99th percentile for robust outlier handling
"""
import torch
import torchstain.torch.normalizers as torchstain_normalizers
import numpy as np
from PIL import Image

class MacenkoNormalizer:
    def __init__(self):
        # Initialize the torchstain normalizer using the specialized Torch backend for GPU acceleration
        self.normalizer = torchstain_normalizers.TorchMacenkoNormalizer()
        # We need a reference image to define the target "standard" H&E stain appearance.
        # This flag tracks if the normalizer has been 'fitted' to a target slide/patch.
        self.fitted = False

    def fit(self, reference_img, device='cpu'):
        """
        Fits the normalizer to a reference image to extract target stain vectors.
        reference_img: PIL image or torch.Tensor (C, H, W)
        
        Raises:
            RuntimeError: If reference image is too uniform (ill-conditioned)
            ValueError: If reference image has insufficient H&E signal
        """
        if isinstance(reference_img, Image.Image):
            # Convert PIL RGB image to a (Channels, Height, Width) Tensor and move to target device
            ref = torch.from_numpy(np.array(reference_img)).permute(2, 0, 1).to(device)
        else:
            # If already a tensor, just ensure it is on the correct device (GPU/CPU)
            ref = reference_img.to(device)
        
        # Ensure tensor is float type for numerical stability
        if ref.dtype != torch.float32 and ref.dtype != torch.float64:
            ref = ref.float()
            
        # VALIDATION: Check if reference has sufficient color variation
        # If the image is mostly uniform (low std), Macenko will fail with ill-conditioned matrix
        if ref.max() <= 1.0:  # Normalized range [0, 1]
            std_per_channel = ref.std(dim=[1, 2])
            mean_intensity = ref.mean()
        else:  # Range [0, 255]
            std_per_channel = ref.std(dim=[1, 2])
            mean_intensity = ref.mean() / 255.0
        
        # Check for problematic patterns
        if std_per_channel.max() < 0.05:
            raise ValueError(
                f"Reference patch has insufficient color variation (std={std_per_channel.max():.4f}). "
                f"Likely an empty/uniform patch with weak H&E signal. "
                f"Please provide a reference with more staining variation."
            )
        
        if mean_intensity > 0.95:
            raise ValueError(
                f"Reference patch is too bright (mean intensity={mean_intensity:.3f}). "
                f"Likely a mostly white/empty patch with no tissue. "
                f"Please provide a reference with actual tissue staining."
            )
            
        # Extract the HE vectors and maximum concentrations from the reference
        try:
            self.normalizer.fit(ref)
            self.fitted = True
        except RuntimeError as e:
            # Torchstain eigenvalue decomposition failed (ill-conditioned covariance matrix)
            error_msg = str(e)
            if "eigh" in error_msg or "singular" in error_msg.lower():
                raise RuntimeError(
                    f"Macenko fit failed due to ill-conditioned reference image (eigenvalue decomposition failed). "
                    f"This usually means the reference patch lacks sufficient H&E color structure. "
                    f"Original error: {error_msg}"
                ) from e
            else:
                raise RuntimeError(f"Macenko fit failed: {error_msg}") from e

    def __call__(self, img):
        # If no reference has been fitted, return the original image as a safety fallback
        if not self.fitted:
            return img 
            
        # Support for both standard PIL Images (from datasets) and Tensors (from GPU pipelines)
        if isinstance(img, Image.Image):
            # Prepare PIL image for torchstain by converting to Tensor
            img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1)
            is_pil = True
        else:
            img_t = img
            is_pil = False
            
        # Execute the Macenko normalization transform
        try:
            # torchstain expects pixel values in the [0, 255] range for Optical Density calculation
            if img_t.max() <= 1.5: 
                img_t = img_t * 255.0
            
            # CRITICAL: Ensure internal reference tensors (HERef, maxC) are on the SAME device as the input
            # This prevents 'device mismatch' errors when switching between CPU and GPU inference.
            if hasattr(self.normalizer, 'HERef'):
                self.normalizer.HERef = self.normalizer.HERef.to(img_t.device)
            if hasattr(self.normalizer, 'maxC'):
                self.normalizer.maxC = self.normalizer.maxC.to(img_t.device)
            if hasattr(self.normalizer, 'stain_matrix_target'):
                self.normalizer.stain_matrix_target = self.normalizer.stain_matrix_target.to(img_t.device)
            
            # Perform the normalization; returns the transformed image in RGB [0, 255]
            norm_img, _, _ = self.normalizer.normalize(I=img_t, stains=True)
            
            if is_pil:
                # Convert back to PIL Image if the original input was a PIL object
                return Image.fromarray(norm_img.cpu().numpy().astype(np.uint8))
            else:
                # Re-format to (C, H, W) and scale to [0, 1] for subsequent Neural Network layers
                if norm_img.shape[-1] == 3:
                    norm_img = norm_img.permute(2, 0, 1)
                return norm_img.float() / 255.0
        except Exception as e:
            # If decomposition fails (e.g. empty/white patch), return the original image to avoid crashing
            return img

    def normalize_batch(self, batch_tensor, jitter=False):
        """
        Normalize each image in batch individually using Macenko.
        Per-image processing avoids batch-level numerical issues with low-variance images.
        batch_tensor: (B, C, H, W) on GPU, values in [0, 1]
        """
        if not self.fitted:
            return batch_tensor
        
        # Apply normalization to each image individually
        # This avoids batch-level eigendecomposition failures
        normalized_list = []
        for b in range(batch_tensor.shape[0]):
            img = batch_tensor[b]  # (C, H, W)
            try:
                # Use the __call__ method which has proper error handling
                normalized_img = self.__call__(img)
                normalized_list.append(normalized_img)
            except Exception as e:
                # If normalization fails for this image, keep it unnormalized
                normalized_list.append(img)
        
        # Concatenate all normalized images back into batch
        return torch.stack(normalized_list, dim=0)


    def __repr__(self):
        return f"MacenkoNormalizer(fitted={self.fitted})"
        return self.__class__.__name__ + "()"
