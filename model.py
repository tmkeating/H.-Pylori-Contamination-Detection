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
    
    # --- Modification Step ---
    # The standard ResNet18 was designed to recognize 1,000 different things (dogs, cars, etc.)
    # We only have 2 classes: "Negative" or "Contaminated".
    # We find out how many connections go into the original final "thinking" layer
    num_ftrs = model.fc.in_features
    # We replace that final layer with a new one that only has 2 outputs
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model # Hand over the complete, customized brain

if __name__ == "__main__":
    # If you run this file directly, it will just show you the structure of the brain
    model = get_model()
    print(model)
