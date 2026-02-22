import torch
import torchstain.torch.normalizers as torchstain_normalizers
import numpy as np
from PIL import Image

class MacenkoNormalizer:
    def __init__(self):
        # Initialize the torchstain normalizer with torch backend
        self.normalizer = torchstain_normalizers.TorchMacenkoNormalizer()
        # We need a reference image. I will provide a standard H&E reference 
        # as a tensor if one isn't provided, but ideally, we fit it to a real patch.
        self.fitted = False

    def fit(self, reference_img, device='cpu'):
        """
        reference_img: PIL image or torch.Tensor (C, H, W)
        """
        if isinstance(reference_img, Image.Image):
            # Convert PIL to Tensor (C, H, W) and move to device
            ref = torch.from_numpy(np.array(reference_img)).permute(2, 0, 1).to(device)
        else:
            ref = reference_img.to(device)
            
        self.normalizer.fit(ref)
        self.fitted = True

    def __call__(self, img):
        if not self.fitted:
            return img # Fallback if not fitted
            
        # Support for both PIL Images (CPU) and Tensors (GPU)
        if isinstance(img, Image.Image):
            img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1)
            is_pil = True
        else:
            img_t = img
            is_pil = False
            
        # Normalize
        try:
            # torchstain expects values in [0, 255]
            if img_t.max() <= 1.5: 
                img_t = img_t * 255.0
            
            # Ensure internal state is on the same device as input
            if hasattr(self.normalizer, 'HERef'):
                self.normalizer.HERef = self.normalizer.HERef.to(img_t.device)
            if hasattr(self.normalizer, 'maxC'):
                self.normalizer.maxC = self.normalizer.maxC.to(img_t.device)
            if hasattr(self.normalizer, 'stain_matrix_target'):
                self.normalizer.stain_matrix_target = self.normalizer.stain_matrix_target.to(img_t.device)
            
            norm_img, _, _ = self.normalizer.normalize(I=img_t, stains=True)
            
            if is_pil:
                return Image.fromarray(norm_img.cpu().numpy().astype(np.uint8))
            else:
                # Ensure it's (C, H, W) for PyTorch transforms
                if norm_img.shape[-1] == 3:
                    norm_img = norm_img.permute(2, 0, 1)
                # Return as float tensor in [0, 1] for next steps
                return norm_img.float() / 255.0
        except Exception as e:
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
        Io = 240.0
        alpha = 1
        beta = 0.15
        
        # 1. Convert whole batch to OD (Vectorized)
        I = (batch_tensor * 255.0).permute(0, 2, 3, 1) # (B, H, W, 3)
        N = H * W
        OD = -torch.log((I.reshape(B, N, 3).float() + 1.0) / Io) # (B, N, 3)
        
        # 2. Batch-Wide Stain Matrix Estimation (Vectorized)
        mask = (OD >= beta).all(dim=-1).float() # (B, N)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=100) # (B, 1)
        
        mean = (OD * mask.unsqueeze(-1)).sum(dim=1) / counts # (B, 3)
        OD_centered = (OD - mean.unsqueeze(1)) * mask.unsqueeze(-1) # (B, N, 3)
        Cov = torch.matmul(OD_centered.transpose(1, 2), OD_centered) / (counts.unsqueeze(-1) - 1).clamp(min=1)
        
        eps = 1e-6 * torch.eye(3, device=device).unsqueeze(0)
        L, V = torch.linalg.eigh(Cov + eps)
        eigvecs = V[:, :, [1, 2]]
        
        That = torch.matmul(OD, eigvecs) # (B, N, 2)
        phi = torch.atan2(That[:, :, 1], That[:, :, 0]) # (B, N)
        phi_masked = torch.where(mask > 0, phi, torch.tensor(float('nan'), device=device))
        
        def batch_nanquantile(x, q):
            x_filled = torch.where(torch.isnan(x), torch.tensor(float('inf'), device=device), x)
            x_sorted, _ = torch.sort(x_filled, dim=1)
            valid_counts = mask.sum(dim=1)
            indices = (q * (valid_counts - 1)).long()
            return x_sorted.gather(1, indices.unsqueeze(1).clamp(min=0)).squeeze(1)

        min_phi = batch_nanquantile(phi_masked, alpha / 100) # (B,)
        max_phi = batch_nanquantile(phi_masked, 1 - alpha / 100) # (B,)
        
        vMin = torch.matmul(eigvecs, torch.stack([torch.cos(min_phi), torch.sin(min_phi)], dim=1).unsqueeze(-1)).squeeze(-1)
        vMax = torch.matmul(eigvecs, torch.stack([torch.cos(max_phi), torch.sin(max_phi)], dim=1).unsqueeze(-1)).squeeze(-1)
        
        condition = (vMin[:, 0] > vMax[:, 0]).unsqueeze(-1).unsqueeze(-1)
        HE_batch = torch.where(condition, torch.stack([vMin, vMax], dim=2), torch.stack([vMax, vMin], dim=2))
        
        ref_HE = self.normalizer.HERef.to(device) # (3, 2)
        is_nan = torch.any(torch.isnan(HE_batch.view(B, -1)), dim=1).unsqueeze(-1).unsqueeze(-1)
        HE_batch = torch.where(is_nan, ref_HE.unsqueeze(0), HE_batch)
        
        # 3. Batch Solve for Concentrations
        Y = OD.permute(0, 2, 1) # (B, 3, N)
        HE_pinv = torch.linalg.pinv(HE_batch) # (B, 2, 3)
        C = torch.matmul(HE_pinv, Y) # (B, 2, N)
        
        # 4. Normalize Concentrations
        def batch_quantile_positive(x, q):
            x_pos = torch.where(x > 0, x, torch.tensor(float('-inf'), device=device))
            x_sorted, _ = torch.sort(x_pos, dim=2)
            valid_counts = (x > 0).sum(dim=2)
            indices = (q * (valid_counts - 1)).long()
            return x_sorted.gather(2, indices.unsqueeze(2).clamp(min=0)).squeeze(2)

        maxC_source = batch_quantile_positive(C, 0.99)
        maxC_source = torch.clamp(maxC_source, min=1e-5)
        
        ref_maxC = self.normalizer.maxCRef.to(device)
        scale = (ref_maxC / maxC_source).unsqueeze(-1) # (B, 2, 1)
        C_norm = C * scale

        # --- PATHOLOGICAL STAIN JITTER (Optional) ---
        # Apply jitter in H&E concentration space instead of RGB ColorJitter
        if jitter:
            # Multiplicative Jitter (alpha): simulate staining intensity variation
            # Sigma 0.2 means +/- 20% intensity shift
            alpha_jitter = (torch.rand(B, 2, 1, device=device) * 0.4) + 0.8
            # Additive Jitter (beta): simulate background noise/wash variation
            beta_jitter = (torch.rand(B, 2, 1, device=device) * 0.1) - 0.05
            C_norm = C_norm * alpha_jitter + beta_jitter

        # 5. Reconstruct and convert back to image space
        OD_norm = torch.matmul(ref_HE, C_norm)
        Inorm = Io * torch.exp(-OD_norm)
        
        Inorm = torch.clamp(Inorm, 0, 255).reshape(B, 3, H, W) / 255.0
        return Inorm.to(dtype)


    def __repr__(self):
        return f"MacenkoNormalizer(fitted={self.fitted})"
        return self.__class__.__name__ + "()"
