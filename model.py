import torch # The core Deep Learning library
import torch.nn as nn # Tools to build layers for our "brain"
from torchvision import models # Access to famous, pre-trained AI architectures

# This function builds the "Artificial Brain" (the model)
def get_model(num_classes=2, pretrained=True):
    # Load a "ResNet18" model. 
    # This is a famous architecture that has already been trained on 1 MILLION images.
    # It already knows how to see basic things like colors, edges, and textures.
    try:
        # Modern way to load the "Already Learned" intelligence (Weights)
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    except ImportError:
        # Older way if your software is a bit behind
        model = models.resnet18(pretrained=pretrained)
    
    # --- Modification Step: The "Sharp" Classification Head ---
    # The standard ResNet18 was designed to recognize 1,000 different things.
    # We replace the final layer with a more specialized, multi-layer "Sharp" head:
    # 1. First Dense Layer (512 -> 256)
    # 2. ReLU Activation (To handle complex visual features)
    # 3. Dropout (0.5) - CRITICAL for preventing overfitting to staining artifacts (Run 51)
    # 4. Final Prediction Layer (256 -> 2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5), 
        nn.Linear(256, num_classes)
    )
    
    return model # Hand over the complete, customized brain

if __name__ == "__main__":
    # If you run this file directly, it will just show you the structure of the brain
    model = get_model()
    print(model)
