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

    def normalize_batch(self, batch_tensor):
        """
        Optimized batch normalization for Macenko.
        batch_tensor: (B, C, H, W) on GPU, values in [0, 1]
        """
        if not self.fitted:
            return batch_tensor
            
        device = batch_tensor.device
        B, C, H, W = batch_tensor.shape
        Io = 240.0
        alpha = 1
        beta = 0.15
        
        # 1. Convert whole batch to OD (Vectorized)
        # torchstain uses: OD = -log((I+1)/Io)
        # batch_tensor is [0,1], scale to [0,255]
        I = (batch_tensor * 255.0).permute(0, 2, 3, 1) # (B, H, W, 3)
        OD = -torch.log((I.reshape(B, -1, 3).float() + 1.0) / Io) # (B, N, 3)
        
        # 2. Estimate HE matrix for each image (This loop is hard to vectorize due to filtering)
        HE_mats = []
        maxCs = []
        
        for i in range(B):
            img_OD = OD[i]
            # remove transparent pixels for this image
            ODhat = img_OD[~torch.any(img_OD < beta, dim=1)]
            
            # Robustness: If too few tissue pixels, use reference
            if ODhat.shape[0] < 100: 
                HE_mats.append(self.normalizer.HERef.to(device))
                continue

            try:
                # compute eigenvectors
                # Use torch.cov with a small diagonal epsilon for stability
                covariance = torch.cov(ODhat.T)
                if torch.any(torch.isnan(covariance)):
                    HE_mats.append(self.normalizer.HERef.to(device))
                    continue
                    
                _, eigvecs = torch.linalg.eigh(covariance)
                eigvecs = eigvecs[:, [1, 2]]
                
                HE = self.normalizer._TorchMacenkoNormalizer__find_HE(ODhat, eigvecs, alpha)
                
                # Check for NaN in HE
                if torch.any(torch.isnan(HE)):
                    HE_mats.append(self.normalizer.HERef.to(device))
                else:
                    HE_mats.append(HE)
            except Exception:
                HE_mats.append(self.normalizer.HERef.to(device))
        
        HE_batch = torch.stack(HE_mats)
        
        # 3. Batch Solve for Concentrations (Vectorized!)
        Y = OD.permute(0, 2, 1)
        
        # Use a more robust solver for potential singular matrices
        # We use the pseudoinverse (pinv) which is numerically more stable 
        # for rank-deficient matrices common in background patches.
        # C = pinv(HE) @ Y
        HE_pinv = torch.linalg.pinv(HE_batch) # (B, 2, 3)
        C = torch.matmul(HE_pinv, Y) # (B, 2, N)
        
        # 4. Normalize Concentrations (Vectorized)
        # Handle zero concentrations to avoid div-by-zero
        maxC_source = torch.quantile(C, 0.99, dim=2)
        maxC_source = torch.clamp(maxC_source, min=1e-5)
        
        # Scaling factor: (ref_maxC / source_maxC)
        # self.normalizer.maxCRef is (2,)
        ref_maxC = self.normalizer.maxCRef.to(device)
        scale = (ref_maxC / maxC_source).unsqueeze(-1) # (B, 2, 1)
        C_norm = C * scale
        
        # 5. Reconstruct (Vectorized!)
        # Inorm = Io * exp(-HERef @ C_norm)
        # HERef is (3, 2)
        ref_HE = self.normalizer.HERef.to(device) # (3, 2)
        
        # (3, 2) @ (B, 2, N) -> (B, 3, N)
        OD_norm = torch.matmul(ref_HE, C_norm)
        Inorm = Io * torch.exp(-OD_norm)
        
        # Final cleanup: clamping and reshaping
        Inorm = torch.clamp(Inorm, 0, 255)
        Inorm = Inorm.reshape(B, 3, H, W) / 255.0
        
        return Inorm.to(batch_tensor.dtype)

    def __repr__(self):
        return f"MacenkoNormalizer(fitted={self.fitted})"
        return self.__class__.__name__ + "()"
