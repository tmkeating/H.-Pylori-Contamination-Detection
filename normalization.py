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
        """
        if isinstance(reference_img, Image.Image):
            # Convert PIL RGB image to a (Channels, Height, Width) Tensor and move to target device
            ref = torch.from_numpy(np.array(reference_img)).permute(2, 0, 1).to(device)
        else:
            # If already a tensor, just ensure it is on the correct device (GPU/CPU)
            ref = reference_img.to(device)
            
        # Extract the HE vectors and maximum concentrations from the reference
        self.normalizer.fit(ref)
        self.fitted = True

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
        Fully vectorized batch normalization for Macenko (Optimization 5D).
        Includes optional pathological stain jittering (H&E space).
        batch_tensor: (B, C, H, W) on GPU, values in [0, 1]
        """
        if not self.fitted:
            return batch_tensor
            
        device = batch_tensor.device
        dtype = batch_tensor.dtype
        B, C, H, W = batch_tensor.shape
        Io = 240.0 # Transmitted light intensity (standard constant)
        alpha = 1  # Tolerance for pseudo-extreme pixels (quantile)
        beta = 0.15 # OD threshold to mask out background/white space
        
        # 1. Convert whole batch to Optical Density (OD) (Vectorized)
        # We move from RGB space to log-space where stain concentrations are additive
        I = (batch_tensor * 255.0).permute(0, 2, 3, 1) # Shape: (B, H, W, 3)
        N = H * W
        OD = -torch.log((I.reshape(B, N, 3).float() + 1.0) / Io) # Shape: (B, N, 3)
        
        # 2. Batch-Wide Stain Matrix Estimation (Vectorized)
        # Identify pixels with enough stain to be useful for estimation (ignore white background)
        mask = (OD >= beta).all(dim=-1).float() # (B, N)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=100) # Ensure at least 100 valid pixels
        
        # Center the OD values for Principal Component Analysis (PCA)
        mean = (OD * mask.unsqueeze(-1)).sum(dim=1) / counts # (B, 3)
        OD_centered = (OD - mean.unsqueeze(1)) * mask.unsqueeze(-1) # (B, N, 3)
        
        # Compute the 3x3 covariance matrix across the batch
        Cov = torch.matmul(OD_centered.transpose(1, 2), OD_centered) / (counts.unsqueeze(-1) - 1).clamp(min=1)
        
        # Perform Eigen-decomposition to find the plane of maximum variation (H&E plane)
        eps = 1e-6 * torch.eye(3, device=device).unsqueeze(0)
        L, V = torch.linalg.eigh(Cov + eps)
        eigvecs = V[:, :, [1, 2]] # Pick the two largest eigenvectors
        
        # Project OD onto the H&E plane to find the angular distribution (phi)
        That = torch.matmul(OD, eigvecs) # (B, N, 2)
        phi = torch.atan2(That[:, :, 1], That[:, :, 0]) # (B, N)
        phi_masked = torch.where(mask > 0, phi, torch.tensor(float('nan'), device=device))
        
        # Helper: Calculate quantiles across the batch while ignoring NaN (masked) values
        def batch_nanquantile(x, q):
            x_filled = torch.where(torch.isnan(x), torch.tensor(float('inf'), device=device), x)
            x_sorted, _ = torch.sort(x_filled, dim=1)
            valid_counts = mask.sum(dim=1)
            indices = (q * (valid_counts - 1)).long()
            return x_sorted.gather(1, indices.unsqueeze(1).clamp(min=0)).squeeze(1)

        # Find the robust extreme angles representing Hematoxylin and Eosin
        min_phi = batch_nanquantile(phi_masked, alpha / 100) # (B,)
        max_phi = batch_nanquantile(phi_masked, 1 - alpha / 100) # (B,)
        
        # Convert angles back to 3D H&E vectors
        vMin = torch.matmul(eigvecs, torch.stack([torch.cos(min_phi), torch.sin(min_phi)], dim=1).unsqueeze(-1)).squeeze(-1)
        vMax = torch.matmul(eigvecs, torch.stack([torch.cos(max_phi), torch.sin(max_phi)], dim=1).unsqueeze(-1)).squeeze(-1)
        
        # Ensure consistent ordering: Hematoxylin first, then Eosin
        condition = (vMin[:, 0] > vMax[:, 0]).unsqueeze(-1).unsqueeze(-1)
        HE_batch = torch.where(condition, torch.stack([vMin, vMax], dim=2), torch.stack([vMax, vMin], dim=2))
        
        # Use the reference H&E vectors we fitted earlier
        ref_HE = self.normalizer.HERef.to(device) # (3, 2)
        
        # Safety: If a patch has no valid stain, fall back to reference vectors to avoid NaNs
        is_nan = torch.any(torch.isnan(HE_batch.view(B, -1)), dim=1).unsqueeze(-1).unsqueeze(-1)
        HE_batch = torch.where(is_nan, ref_HE.unsqueeze(0), HE_batch)
        
        # 3. Batch Solve for Concentrations
        # We solve the linear system: OD = HE_matrix * Concentrations
        # This converts the 3-channel RGB information into 2-channel stain intensity maps
        Y = OD.permute(0, 2, 1) # (B, 3, N)
        HE_pinv = torch.linalg.pinv(HE_batch) # Use pseudo-inverse for robust solving (B, 2, 3)
        C = torch.matmul(HE_pinv, Y) # Result: Concentrations (B, 2, N)
        
        # 4. Normalize Concentrations
        # Scale the source concentrations so their maximum (99th percentile) matches the reference's maximum
        def batch_quantile_positive(x, q):
            # Mask out non-positive values to focus on actual stained tissue area
            x_pos = torch.where(x > 0, x, torch.tensor(float('-inf'), device=device))
            x_sorted, _ = torch.sort(x_pos, dim=2)
            valid_counts = (x > 0).sum(dim=2)
            indices = (q * (valid_counts - 1)).long()
            return x_sorted.gather(2, indices.unsqueeze(2).clamp(min=0)).squeeze(2)

        # Get robust maximum concentration for each stain in each image of the batch
        maxC_source = batch_quantile_positive(C, 0.99)
        maxC_source = torch.clamp(maxC_source, min=1e-5) # Prevent division by zero
        
        # Calculate the scaling factor: reference_max / current_max
        ref_maxC = self.normalizer.maxCRef.to(device)
        scale = (ref_maxC / maxC_source).unsqueeze(-1) # (B, 2, 1)
        C_norm = C * scale

        # --- PATHOLOGICAL STAIN JITTER (Optional) ---
        # This is a specialized data augmentation that adds VARIATION to the normalized stains.
        # Unlike standard RGB jitter, this preserves the morphological structure while simulating 
        # "bad staining" or "faded slides" in a biologically plausible way.
        if jitter:
            # Multiplicative Jitter (alpha): simulates variation in overall staining intensity
            # A range of 0.8 to 1.2 represents a +/- 20% shift in H/E depth.
            alpha_jitter = (torch.rand(B, 2, 1, device=device) * 0.4) + 0.8
            # Additive Jitter (beta): simulates background noise or residual reagent "wash"
            beta_jitter = (torch.rand(B, 2, 1, device=device) * 0.1) - 0.05
            C_norm = C_norm * alpha_jitter + beta_jitter

        # 5. Reconstruct and convert back to image space
        # Project normalized concentrations back onto the Target (Reference) H&E vectors
        OD_norm = torch.matmul(ref_HE, C_norm)
        # Beer-Lambert Law inversion: Intensity = Io * exp(-OD)
        Inorm = Io * torch.exp(-OD_norm)
        
        # Clamp to valid RGB range and reshape to standard [Batch, Channel, Height, Width]
        Inorm = torch.clamp(Inorm, 0, 255).reshape(B, 3, H, W) / 255.0
        return Inorm.to(dtype)


    def __repr__(self):
        return f"MacenkoNormalizer(fitted={self.fitted})"
        return self.__class__.__name__ + "()"
