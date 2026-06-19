"""
Domain Adversarial Neural Networks (DANN) for Experiment-Invariant Feature Learning.

PURPOSE:
--------
Implements gradient reversal and adversary heads to prevent models from learning
experiment-specific staining artifacts during DeepHP H&E pre-training. Forces the
backbone to learn features that don't encode experiment identity, improving
generalization across different H&E staining protocols and scanners.

HOW IT WORKS:
-------------
- AdversaryHead: Predicts experiment ID (0-32) from learned features
- GradientReversalLayer: Negates gradients during backprop to "confuse" adversary
- Result: Feature extractor learns representations that the adversary cannot use
- Loss: Both classification loss (predict H. pylori) + adversary loss (predict experiment)
  Total loss = classification_loss + dann_weight * adversary_loss

USAGE IN TRAINING:
------------------
from domain_adversarial import GradientReversalLayer, AdversaryHead

# During model initialization:
grad_rev = GradientReversalLayer(lambda_=1.0)
adversary = AdversaryHead(feature_dim=768, num_experiments=33)

# During training loop (dataset returns 3-tuple: image, label, exp_idx):
features = backbone(images)                    # (B, 768)
class_logits = head(features)                  # (B, 2)
class_loss = criterion(class_logits, labels)

# DANN loss computation (features go through gradient reversal):
reversed_features = grad_rev(features)         # (B, 768), gradients will be negated
exp_logits = adversary(reversed_features)      # (B, 33)
dann_loss = criterion(exp_logits, exp_indices)

total_loss = class_loss + dann_weight * dann_loss
total_loss.backward()  # Gradients negated for reversed_features path

REFERENCES:
-----------
- Ganin & Lempitsky (2015): "Unsupervised Domain Adaptation by Backpropagation"
- Paper: https://arxiv.org/abs/1409.7495
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversal(Function):
    """
    Gradient reversal layer that negates gradients during backprop.
    
    During forward pass: passes data through unchanged
    During backward pass: reverses the gradient sign
    
    This forces the feature extractor to learn representations that the
    adversary head cannot use to predict the domain (experiment).
    
    Reference: Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation" (2015)
    """
    
    @staticmethod
    def forward(ctx, x, lambda_: 1.0):
        """
        Forward pass: just return the input unchanged.
        Store lambda for use in backward.
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: negate gradients and scale by lambda.
        """
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Module wrapper for gradient reversal.
    
    Usage:
        >>> grad_rev = GradientReversalLayer(lambda_=1.0)
        >>> x = torch.randn(32, 768)
        >>> x_reversed = grad_rev(x)  # Forward: x unchanged
        >>> loss.backward()  # Backward: gradients negated
    """
    
    def __init__(self, lambda_: float = 1.0):
        """
        Args:
            lambda_: Scaling factor for reversed gradients. Higher values
                    give more weight to adversary loss during backprop.
        """
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversal.apply(x, self.lambda_)


class AdversaryHead(nn.Module):
    """
    Experiment/domain adversary head.
    
    Takes learned features and tries to predict which experiment the sample
    came from. This head is trained to maximize loss (via gradient reversal),
    forcing the feature extractor to be experiment-agnostic.
    
    Architecture:
        - Input: (B, feature_dim) feature vectors from backbone
        - Hidden layers: Progressively reduced to num_experiments
        - Output: (B, num_experiments) logits for experiment classification
    """
    
    def __init__(self, feature_dim: int, num_experiments: int, hidden_dim: int = 256):
        """
        Args:
            feature_dim: Dimension of input features (e.g., 768 for ConvNeXt-Tiny)
            num_experiments: Number of unique experiments in the dataset
            hidden_dim: Size of hidden layers
        """
        super(AdversaryHead, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim // 2, num_experiments),
        )
    
    def forward(self, features):
        """
        Args:
            features: (B, feature_dim) tensor of features from backbone
        
        Returns:
            logits: (B, num_experiments) logits for experiment prediction
        """
        return self.net(features)


def add_adversary_to_model(model, feature_dim: int, num_experiments: int, 
                           lambda_: float = 1.0, hidden_dim: int = 256):
    """
    Attach gradient reversal and adversary head to an existing model.
    
    Args:
        model: Backbone model (e.g., ConvNeXt)
        feature_dim: Dimension of features produced by backbone
        num_experiments: Number of unique experiments
        lambda_: Gradient reversal scaling factor
        hidden_dim: Hidden dimension for adversary head
    
    Returns:
        tuple: (gradient_reversal_layer, adversary_head)
    
    Example:
        >>> model = create_backbone()
        >>> grad_rev, adversary = add_adversary_to_model(model, 768, 33)
        >>> features = model(images)  # (B, 768)
        >>> features_reversed = grad_rev(features)
        >>> exp_logits = adversary(features_reversed)
    """
    grad_rev = GradientReversalLayer(lambda_=lambda_)
    adversary = AdversaryHead(feature_dim, num_experiments, hidden_dim)
    
    return grad_rev, adversary
