import torch # The core Deep Learning library
import torch.nn as nn # Tools to build layers for our "brain"
from torchvision import models # Access to famous, pre-trained AI architectures

class AttentionGate(nn.Module):
    """
    Attention mechanism for Multiple Instance Learning (MIL).
    Learns to weigh patches based on their diagnostic relevance.
    """
    def __init__(self, feature_dim=2048, hidden_dim=256):
        super(AttentionGate, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: (N, feature_dim)
        A = self.attention(x) # (N, 1)
        A = torch.transpose(A, 1, 0) # (1, N)
        A = nn.functional.softmax(A, dim=1) # softmax over patches
        return A

class HPyNet(nn.Module):
    """
    Unified Wrapper for modern architectures (ResNet50 or ConvNeXt).
    - ResNet50: Classic, fast, highly optimized for A40 (torch.compile).
    - ConvNeXt-Tiny: Modern, 7x7 kernels, better morphology extraction.
    """
    def __init__(self, model_name="resnet50", num_classes=2, pretrained=True):
        super(HPyNet, self).__init__()
        self.model_name = model_name.lower()
        
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
        # Dynamically sized based on backbone output
        self.patch_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 3. MIL Attention Gate (Iteration 4 Readiness)
        self.attention_gate = AttentionGate(feature_dim=self.feature_dim)

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

        # 2. Attention aggregation
        # features shape: (Bag_Size, feature_dim)
        A = self.attention_gate(features) # (1, Bag_Size)
        M = torch.mm(A, features)         # (1, feature_dim)
        
        # 3. Final classification on the aggregated feature
        logits = self.patch_head(M)             # (1, num_classes)
        return logits, A

# Factory function to build the model
def get_model(model_name="resnet50", num_classes=2, pretrained=True):
    return HPyNet(model_name=model_name, num_classes=num_classes, pretrained=pretrained)

if __name__ == "__main__":
    # If you run this file directly, it will just show you the structure of the brain
    model = get_model()
    print(model)
