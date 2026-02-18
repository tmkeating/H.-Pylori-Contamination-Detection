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
    Wrapper for ResNet50 with integrated Attention-MIL capability.
    Can operate in Patch Mode (standard ResNet) or Bag Mode (Attention).
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(HPyNet, self).__init__()
        try:
            from torchvision.models import ResNet50_Weights
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        except ImportError:
            self.backbone = models.resnet50(pretrained=pretrained)
        
        self.feature_dim = self.backbone.fc.in_features
        
        # Keep the standard head for patch-level training (Iteration 2)
        self.patch_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # MIL Attention Gate
        self.attention_gate = AttentionGate(feature_dim=self.feature_dim)
        
        # Replace backbone fc with identity to extract features easily
        self.backbone.fc = nn.Identity()

    def forward(self, x, return_features=False):
        # Standard patch-level forward pass
        features = self.backbone(x)
        logits = self.patch_head(features)
        
        if return_features:
            return logits, features
        return logits

    def forward_bag(self, batch_features):
        """
        Aggregates multiple patches into a single patient-level diagnosis
        using the learned attention weights.
        """
        # batch_features: (N_patches, 2048)
        A = self.attention_gate(batch_features) # (1, N_patches)
        M = torch.mm(A, batch_features) # (1, 2048) - Weighted feature vector
        logits = self.patch_head(M) # (1, num_classes)
        return logits, A

# This function builds the "Artificial Brain" (the model)
def get_model(num_classes=2, pretrained=True):
    return HPyNet(num_classes=num_classes, pretrained=pretrained)

if __name__ == "__main__":
    # If you run this file directly, it will just show you the structure of the brain
    model = get_model()
    print(model)
