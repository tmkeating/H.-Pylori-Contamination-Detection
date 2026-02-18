import torch # The core Deep Learning library
import torch.nn as nn # Tools to build layers for our "brain"
from torchvision import models # Access to famous, pre-trained AI architectures

# This function builds the "Artificial Brain" (the model)
def get_model(num_classes=2, pretrained=True):
    # Load a "ResNet50" model (Iteration 2 Upgrade)
    # This is a deeper architecture than ResNet18, providing 2048 features
    # for more fine-grained morphological analysis.
    try:
        from torchvision.models import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    except ImportError:
        model = models.resnet50(pretrained=pretrained)
    
    # --- Modification Step: The "Powerhouse" Classification Head ---
    # We replace the final layer with a multi-layer head to handle the 2048-D features:
    # 1. Linear(2048, 512)
    # 2. ReLU Activation
    # 3. Dropout (0.5) - Essential for regularizing the higher-capacity ResNet50
    # 4. Final Prediction Layer (512 -> 2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model # Hand over the complete, customized brain

if __name__ == "__main__":
    # If you run this file directly, it will just show you the structure of the brain
    model = get_model()
    print(model)
