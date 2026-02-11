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

    def fit(self, reference_img):
        """
        reference_img: PIL image or torch.Tensor (C, H, W)
        """
        if isinstance(reference_img, Image.Image):
            # Convert PIL to Tensor (C, H, W)
            ref = torch.from_numpy(np.array(reference_img)).permute(2, 0, 1)
        else:
            ref = reference_img
            
        self.normalizer.fit(ref)
        self.fitted = True

    def __call__(self, img):
        if not self.fitted:
            return img # Fallback if not fitted
            
        # torchstain expects (C, H, W) or (H, W, C) depending on backend
        # For torch backend, it usually expects (C, H, W) in range [0, 255]
        if isinstance(img, Image.Image):
            img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        else:
            img_t = img
            
        # Normalize
        try:
            norm_img, _, _ = self.normalizer.normalize(I=img_t, stains=True)
            # norm_img is returned as torch.uint8 (C, H, W)
            return Image.fromarray(norm_img.permute(1, 2, 0).numpy())
        except Exception as e:
            # Sometimes Macenko fails on white patches or weird artifacts
            return img

    def __repr__(self):
        return f"MacenkoNormalizer(fitted={self.fitted})"

    def __repr__(self):
        return self.__class__.__name__ + "()"
