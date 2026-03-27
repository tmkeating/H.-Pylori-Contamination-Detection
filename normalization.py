"""
# H. Pylori Staining Normalization Utility
# ----------------------------------------
# This script implements Macenko normalization to standardize histological staining
# across different whole slide images (WSIs). This reduces the model's sensitivity 
# to staining intensity variations (H&E depth) between labs or tissue blocks.
#
# What it does:
#   1. Fits a Macenko normalizer to a reference H&E image.
#   2. Decomposes the stain matrix of input patches.
#   3. Projects input patches onto the reference stain space.
#   4. Supports both CPU (PIL/Numpy) and GPU (Torch) backends for high-speed inference.
#
# Usage:
#   normalizer = MacenkoNormalizer()
#   normalizer.fit(reference_patch)
#   normalized_patch = normalizer(input_patch)
# ----------------------------------------
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
