#!/usr/bin/env python3
"""
DeepHP Model Wrapper with Per-Fold Threshold Application

Wraps the trained backbone model with fold-specific thresholds for proper
classification. This wrapper can be saved alongside the .pth file.

Usage:
    python3 deephp_backbone_with_threshold.py --run 01_34.1
    python3 deephp_backbone_with_threshold.py --run 01_34.2 --model convnext_tiny

Loads thresholds from: {run_id}_calibrated_thresholds.json
"""

import torch
import torch.nn as nn
import json
import argparse
from pathlib import Path


class DeepHPBackboneWithThreshold(nn.Module):
    """
    Wrapper around trained DeepHP backbone that applies fold-specific thresholds.
    
    Example:
        # Load backbone model
        backbone = torch.load('convnext_tiny_f0.pth')
        
        # Load thresholds
        with open('calibrated_thresholds.json') as f:
            thresholds_data = json.load(f)
        
        # Wrap with threshold application
        model = DeepHPBackboneWithThreshold(backbone, thresholds_data, fold_idx=0)
        
        # Forward pass now outputs thresholded predictions (0/1)
        predictions = model(images)  # shape: [batch, 1] with values 0 or 1
    """
    
    def __init__(self, backbone, thresholds_data, fold_idx):
        """
        Args:
            backbone: Trained model that outputs logits [batch, 2]
            thresholds_data: Dict with 'fold_thresholds' key containing per-fold info
            fold_idx: Which fold's threshold to use (0-4)
        """
        super().__init__()
        self.backbone = backbone
        self.fold_idx = fold_idx
        
        # Extract threshold for this fold
        fold_key = str(fold_idx)
        if fold_key in thresholds_data.get('fold_thresholds', {}):
            self.threshold = thresholds_data['fold_thresholds'][fold_key]['threshold']
        else:
            print(f"Warning: No threshold found for fold {fold_idx}, using 0.5")
            self.threshold = 0.5
    
    def forward(self, x, return_logits=False, return_probabilities=False):
        """
        Forward pass with threshold application.
        
        Args:
            x: Input images [batch, C, H, W]
            return_logits: If True, also return raw logits
            return_probabilities: If True, also return probabilities before thresholding
        
        Returns:
            predictions: [batch, 1] with values 0 or 1 (after thresholding)
            logits: (optional) [batch, 2] raw model outputs
            probabilities: (optional) [batch] probabilities for positive class
        """
        # Get logits from backbone
        logits = self.backbone(x)  # [batch, 2]
        
        # Extract positive class probability
        probabilities = torch.softmax(logits, dim=1)[:, 1]  # [batch]
        
        # Apply fold-specific threshold
        predictions = (probabilities >= self.threshold).long()  # [batch] with 0/1
        
        # Return based on request
        if return_logits and return_probabilities:
            return predictions, logits, probabilities
        elif return_logits:
            return predictions, logits
        elif return_probabilities:
            return predictions, probabilities
        else:
            return predictions
    
    def get_probabilities(self, x):
        """Get raw probabilities without thresholding (for visualization/analysis)."""
        logits = self.backbone(x)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        return probabilities
    
    def get_threshold(self):
        """Get the threshold being used for this fold."""
        return self.threshold


def example_usage(run_id='01_34.1', model_name='convnext_tiny'):
    """
    Example: Load backbone models with calibrated thresholds
    
    Args:
        run_id: Run identifier with iteration (e.g., '01_34.1')
        model_name: Model name (default: 'convnext_tiny')
    """
    import glob
    
    # Load thresholds
    threshold_file = f'/home/tkeating/model/H.-Pylori-Contamination-Detection/results/{run_id}_calibrated_thresholds.json'
    with open(threshold_file) as f:
        thresholds_data = json.load(f)
    
    print("DeepHP Backbones with Calibrated Thresholds")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Model: {model_name}")
    print("=" * 80)
    
    # Load all fold models
    fold_models = {}
    for fold_idx in range(5):
        # Pattern: *f{fold}_convnext_tiny_model_brain.pth
        model_files = glob.glob(f'/home/tkeating/model/H.-Pylori-Contamination-Detection/results/*f{fold_idx}_{model_name}_model_brain.pth')
        if model_files:
            backbone_path = model_files[0]
            backbone = torch.load(backbone_path, map_location='cpu')
            
            # Wrap with threshold
            wrapped_model = DeepHPBackboneWithThreshold(backbone, thresholds_data, fold_idx)
            fold_models[fold_idx] = wrapped_model
            
            threshold = wrapped_model.get_threshold()
            print(f"Fold {fold_idx}: threshold={threshold:.3f} (from {Path(backbone_path).name})")
    
    print("=" * 80)
    print(f"\nLoaded {len(fold_models)} fold models with calibrated thresholds")
    print("\nUsage in ensemble:")
    print("  predictions = [model(batch) for model in fold_models.values()]")
    print("  ensemble_pred = torch.cat(predictions, dim=0).mode(dim=0)[0]  # majority vote")


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(description='Load DeepHP backbones with calibrated thresholds')
    parser.add_argument('--run', default='01_34.1', help='Run ID (e.g., 01_34.1)')
    parser.add_argument('--model', default='convnext_tiny', help='Model name')
    args = parser.parse_args()
    
    example_usage(run_id=args.run, model_name=args.model)


if __name__ == '__main__':
    main()
