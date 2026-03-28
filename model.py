"""
H. Pylori Contamination Detection - Neural Network Architecture (HPyNet)
========================================================================

OVERVIEW
--------
This module implements the complete deep learning architecture for H. Pylori detection
in high-resolution histological whole-slide images (WSI). The pipeline combines:
  - Modern backbone architectures (ConvNeXt-Tiny or ResNet50) for feature extraction
  - Gated Attention Multiple Instance Learning (MIL) for patient-level diagnosis
  - Memory-efficient inference via gradient checkpointing and chunked processing
  - Temperature-scaled attention for interpretability and clinical deployment

PURPOSE
-------
Provides a unified model interface for both training and inference on patch bags.
Supports:
  - Patch-level prediction (single images or small batches)
  - Bag-level aggregation (100s-1000s of patches from a single patient)
  - Train and eval modes for proper handling of stochastic components
  - Clinical deployment modes (Searcher vs. Auditor profiles via temperature tuning)

ARCHITECTURE
------------

TWO-STAGE DESIGN:
  Stage 1 (Backbone): Feature extraction from individual patches
    - Input: (B, 3, 448, 448) preprocessed histology images
    - Output: (B, feature_dim) learned feature vectors
    - Architectures: ConvNeXt-Tiny (768-dim) or ResNet50 (2048-dim)
    - Pretrained on ImageNet → fine-tuned for histology domain

  Stage 2 (MIL Head): Learns from patch-level features to make patient-level predictions
    - Input: (Bag_Size, feature_dim) aggregated features from all patches
    - Attention: Gated mechanism learns which patches are diagnostically relevant
    - Classification: Deep head processes attention-weighted features
    - Output: (1, num_classes) patient-level logits

GATED ATTENTION MECHANISM:
  Addresses the challenge of sparse bacterial detection in "noisy" backgrounds (stain
  artifacts, cell debris, tissue mimics).

  Dual-pathway design:
    V-pathway (tanh): Captures non-linear morphological interactions
    U-pathway (sigmoid): Noise filtering gate (learned to suppress false signals)
    Final score: Element-wise multiplication (V ⊙ U) → softmax

  Temperature scaling: Allows tuning of attention entropy
    - Lower T: Focus on single most-suspicious patch (Searcher mode)
    - T = 1.0: Balanced attention (standard training)
    - Higher T: Diffuse attention (Auditor mode for consensus)

HOW IT WORKS
------------

TRAINING FLOW (patch-level):
  1. Load mini-batch of patches: (B, 3, 448, 448)
  2. Backbone forward: Extract features → (B, feature_dim)
  3. Classification head: Compute patch logits → (B, 2)
  4. Loss: Focal Loss to handle false negatives
  5. Backward: Standard SGD/Adam updates

INFERENCE FLOW (bag-level, patient diagnosis):
  1. Load all patches for patient: (Bag_Size, 3, 448, 448)
  2. Memory-efficient processing:
     a. Chunked feature extraction (prevent OOM on 1000-patch bags)
     b. Gradient checkpointing (recompute activations during backward)
  3. MIL aggregation:
     a. Compute attention weights: A ∈ (1, Bag_Size)
     b. Weighted pooling: M = A @ Features
  4. Final classification: logits = head(M)
  5. Patient probability: prob = softmax(logits)[1]

TEMPERATURE SCALING:
  During inference, adjust attention sharpness:
    model.attention_gate.temperature = torch.tensor([0.5])  # Searcher (focused)
    model.attention_gate.temperature = torch.tensor([1.0])  # Standard
    model.attention_gate.temperature = torch.tensor([2.0])  # Auditor (diffuse)

USAGE
-----

BASIC USAGE:

  from model import get_model

  # Create model
  model = get_model(
      model_name='convnext_tiny',  # or 'resnet50'
      num_classes=2,
      pretrained=True,
      pool_type='attention'         # or 'max' for max-pooling baseline
  ).to(device)

PATCH-LEVEL INFERENCE:

  # Single forward pass on a batch of images
  batch_images = torch.randn(32, 3, 448, 448).to(device)
  logits = model(batch_images)              # (32, 2)
  probs = F.softmax(logits, dim=1)          # (32, 2) with softmax

  # Get features for gradient computation (e.g., Grad-CAM)
  logits, features = model(batch_images, return_features=True)
  # logits: (32, 2), features: (32, feature_dim)

BAG-LEVEL INFERENCE (Patient Diagnosis):

  # MIL forward pass on all patches from a patient
  patient_patches = torch.randn(847, 3, 448, 448).to(device)
  logits, attention = model.forward_bag(patient_patches, chunk_size=64)
  # logits: (1, 2) - patient-level prediction
  # attention: (1, 847) - which patches matter most

TRAINING MODE:

  model.train()
  # Mini-batch training on patches (not full bags)
  for batch_images, batch_labels in train_loader:
      logits = model(batch_images)
      loss = focal_loss(logits, batch_labels)
      loss.backward()
      optimizer.step()

CLASS REFERENCE
---------------

AttentionGate(feature_dim=2048, hidden_dim=256)
  Gated Attention mechanism for MIL pooling.
  
  Components:
    - v_proj: Non-linear projection (tanh) for morphology interaction
    - u_gate: Learned sigmoid gate for noise filtering
    - w_score: Final attention score computation
    - temperature: Learnable parameter for entropy control
  
  Forward:
    Input: (N, feature_dim) patch features
    Output: (1, N) normalized attention weights (softmax)
  
  Attributes:
    - temperature: nn.Parameter for attention sharpness (tune during inference)

HPyNet(model_name, num_classes, pretrained, pool_type)
  Main model wrapper combining backbone + MIL head.
  
  Parameters:
    model_name: "resnet50" or "convnext_tiny"
    num_classes: 2 for binary classification (negative/positive)
    pretrained: Load ImageNet weights (highly recommended)
    pool_type: "attention" (learnable) or "max" (baseline)
  
  Methods:
    forward(x, return_features=False)
      Patch-level forward pass
      Input: (B, 3, 448, 448)
      Output: (B, num_classes) logits or (B, num_classes), (B, feature_dim) if return_features
      
    forward_bag(x, chunk_size=8)
      Memory-efficient bag-level aggregation for MIL
      Input: (Bag_Size, 3, 448, 448)
      Output: ((1, num_classes), (1, Bag_Size)) logits and attention weights
      Uses gradient checkpointing for memory efficiency
  
  Attributes:
    backbone: Feature extractor (ResNet50 or ConvNeXt-Tiny)
    patch_head: Classification head (Linear layers with ReLU/Dropout)
    attention_gate: AttentionGate instance (None if pool_type="max")
    feature_dim: Dimensionality of backbone output (768 or 2048)
    pool_type: Aggregation method used in forward_bag

get_model(model_name, num_classes, pretrained, pool_type)
  Factory function to instantiate model.
  Recommended: Use this instead of direct class instantiation.
  
  Returns: HPyNet instance ready for training/inference

MODEL VARIANTS
--------------

CONVNEXT-TINY (Recommended for H. Pylori)
  - 28M parameters, modern architecture
  - 7×7 receptive fields → better morphology extraction
  - Faster training, fewer GPU hours
  - Default backbone in production

RESNET50 (Classical Baseline)
  - 25M parameters, extensively tested
  - Highly optimized for torch.compile
  - Stable training, good generalization
  - Useful for cross-architecture validation

POOLING STRATEGIES

ATTENTION (Default)
  - Learnable: Can adapt to patient cohort
  - Interpretable: Attention weights provide diagnostic hints
  - Flexible: Temperature tuning for deployment profiles
  - Recommended for clinical deployment

MAX-POOLING (Baseline)
  - Non-learnable: Fixed aggregation strategy
  - Fast: No attention gate computation
  - Precision-focused: Routes gradients only to top feature
  - Use for: Ablation studies, extreme resource constraints

INTEGRATION POINTS
------------------
Called from:
  - train.py: Main training loop, model instantiation and checkpoint saving
  - generate_visuals.py: Loading trained checkpoint for evaluation reporting
  - Custom scripts: Inference and model analysis

DEPENDENCIES
------------
  - PyTorch: nn.Module, backprop, gradient checkpointing
  - Torchvision: ResNet50, ConvNeXt-Tiny pretrained weights
  - CUDA (optional but recommended): GPU acceleration for training/inference

NOTES
-----
  - All forward passes expect (B, 3, 448, 448) images (or (Bag_Size, 3, 448, 448) for bags)
  - Images should be normalized to ImageNet stats: μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]
  - Gradient checkpointing in forward_bag trades compute for GPU memory (recommended for bags > 500 patches)
  - Chunk size in forward_bag should be tuned to GPU VRAM (larger chunks = faster, 64-256 typical)
  - Temperature scaling should be adjusted post-training for clinical deployment (not during training)
  - Frozen batch norm (if enabled in training) prevents noise floor instability
  - Max-pooling baseline ignores attention gate (set to None internally)
  - Output features from backbone are guaranteed to be flattened to 2D before classification head
"""
import torch # The core Deep Learning library
import torch.nn as nn # Tools to build layers for our "brain"
from torchvision import models # Access to famous, pre-trained AI architectures

class AttentionGate(nn.Module):
    """
    Gated Attention mechanism for Multiple Instance Learning (MIL).
    This layer learns to weigh each patch in a Whole Slide Image (WSI) bag 
    proportionally to its diagnostic relevance.

    TECHNICAL RATIONALE: tanh(V) * sigmoid(U) Gating
    -----------------------------------------------
    Standard attention often struggles with "mimics" (stain precipitate or 
    cell debris) that visually resemble H. Pylori. 
    - tanh(Vx): Captures non-linear feature interactions and morphology.
    - sigmoid(Ux): Acts as a learned noise gate. If a patch contains 
      stain precipitate, the model learns to output a low sigmoid value 
      for that patch, effectively "muting" the noise before it reaches 
      the final classifier.

    CLINICAL SAFETY: Attention Temperature (T=1.0)
    ----------------------------------------------
    The temperature parameter controls the entropy of the decision. 
    - T=1.0: Initial state for balanced search.
    - During inference, a 'Searcher' profile might use lower T to 
      peak on the single most suspicious patch (maximizing sensitivity).
    - An 'Auditor' profile uses higher T to integrate context across 
      multiple patches, reducing the impact of isolated artifacts.
    """
    def __init__(self, feature_dim=2048, hidden_dim=256):
        super(AttentionGate, self).__init__()
        # V: Non-linear projection
        self.v_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh()
        )
        # U: Noise Filtering Gate
        self.u_gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid()
        )
        # Final Attention Score
        self.w_score = nn.Linear(hidden_dim, 1)
        
        # Auditor Initialization: Standard Temperature for Stable Aggregation
        # Initialized to 1.0; SWA will handle the flattening logic during training.
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x shape: (N, feature_dim)
        v = self.v_proj(x) # (N, hidden_dim)
        u = self.u_gate(x) # (N, hidden_dim)
        
        # Gated interaction: Element-wise multiplication
        gated = v * u # (N, hidden_dim)
        
        A = self.w_score(gated) # (N, 1)
        A = A / self.temperature # Apply temperature scaling
        A = torch.transpose(A, 1, 0) # (1, N)
        A = nn.functional.softmax(A, dim=1) # softmax over patches
        return A

class HPyNet(nn.Module):
    """
    Unified Wrapper for modern architectures (ResNet50 or ConvNeXt).
    - ResNet50: Classic, fast, highly optimized for A40 (torch.compile).
    - ConvNeXt-Tiny: Modern, 7x7 kernels, better morphology extraction.
    """
    def __init__(self, model_name="resnet50", num_classes=2, pretrained=True, pool_type="attention"):
        super(HPyNet, self).__init__()
        self.model_name = model_name.lower()
        self.pool_type = pool_type.lower()
        
        # 1. Initialize Backbone
        if self.model_name == "resnet50":
            try:
                from torchvision.models import ResNet50_Weights
                self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            except ImportError:
                self.backbone = models.resnet50(pretrained=pretrained)
            
            self.feature_dim = self.backbone.fc.in_features # 2048
            self.backbone.fc = nn.Identity()
            
        elif self.model_name == "convnext_tiny":
            try:
                from torchvision.models import ConvNeXt_Tiny_Weights
                self.backbone = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
            except ImportError:
                # Older torchvision versions might not support ConvNeXt via weights enum
                self.backbone = models.convnext_tiny(pretrained=pretrained)
            
            # ConvNeXt classification head is at .classifier[2]
            self.feature_dim = self.backbone.classifier[2].in_features # 768
            self.backbone.classifier[2] = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {model_name}. Use 'resnet50' or 'convnext_tiny'.")

        # 2. Universal Deep Classification Head
        # Restored Auditor Configuration: Standard Dropout for Generalization
        self.patch_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Standard 0.5 for robust feature selection
            nn.Linear(512, num_classes)
        )
        
        # 3. MIL Attention Gate (Iteration 4 Readiness)
        if self.pool_type == "attention":
            self.attention_gate = AttentionGate(feature_dim=self.feature_dim)
        elif self.pool_type == "max":
            self.attention_gate = None # Max-pooling doesn't need a learnable gate
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}. Use 'attention' or 'max'.")

    def forward(self, x, return_features=False):
        # Standard patch-level forward pass
        features = self.backbone(x)
        
        # ConvNeXt already applies global average pooling internally before the classifier block,
        # but ResNet50 Identity replaces the layer after adaptive pooling.
        # This wrapper ensures a 2D feature vector (Batch, Dim) reaches the head.
        if len(features.shape) > 2:
             features = torch.flatten(features, 1)
             
        logits = self.patch_head(features)
        
        if return_features:
            return logits, features
        return logits

    def forward_bag(self, x, chunk_size=8):
        """
        Memory-efficient feature extraction for large MIL bags.
        x: Input tensor (Bag_Size, C, H, W)
        """
        # 1. Chunked Feature Extraction to prevent CUDA OOM
        bag_size = x.size(0)
        all_features = []
        
        # Use Gradient Checkpointing to save memory during backbone forward pass
        from torch.utils.checkpoint import checkpoint
        
        for i in range(0, bag_size, chunk_size):
            chunk = x[i:i + chunk_size]
            
            # checkpointing the backbone call
            # This trades compute for memory by re-calculating activations during backward
            if self.training:
                feat = checkpoint(self.backbone, chunk, use_reentrant=False)
            else:
                feat = self.backbone(chunk)
            
            if len(feat.shape) > 2:
                feat = torch.flatten(feat, 1)
            all_features.append(feat)
            
        features = torch.cat(all_features, dim=0)

        # 2. Pooling aggregation
        # features shape: (Bag_Size, feature_dim)
        if self.pool_type == "attention":
            A = self.attention_gate(features) # (1, Bag_Size)
            M = torch.mm(A, features)         # (1, feature_dim)
        else:
            # Global Max Pooling (Iteration 22: Precision Searcher)
            # Route gradients ONLY to the most suspicious feature
            M, _ = torch.max(features, dim=0, keepdim=True) # (1, feature_dim)
            A = None # No attention weights in max-pooling mode
        
        # 3. Final classification on the aggregated feature
        logits = self.patch_head(M)             # (1, num_classes)
        return logits, A

# Factory function to build the model
def get_model(model_name="resnet50", num_classes=2, pretrained=True, pool_type="attention"):
    return HPyNet(model_name=model_name, num_classes=num_classes, pretrained=pretrained, pool_type=pool_type)

if __name__ == "__main__":
    # If you run this file directly, it will just show you the structure of the brain
    model = get_model()
    print(model)
