
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from dataset import HPyloriDataset
from model import get_model
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, 
    precision_recall_curve, average_precision_score, classification_report
)
from torchvision import transforms
from PIL import Image

# --- Config ---
# These paths are set through command line arguments
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Paths
BASE_PATH = "/import/fhome/vlia/HelicoDataSet"
if not os.path.exists(BASE_PATH):
    BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), "..", "HelicoDataSet"))

PATIENT_CSV = os.path.join(BASE_PATH, "PatientDiagnosis.csv")
PATCH_CSV = os.path.join(BASE_PATH, "HP_WSI-CoordAnnotatedAllPatches.xlsx")
TRAIN_DIR = os.path.join(BASE_PATH, "CrossValidation/Annotated")

# Preprocessing (Deterministic for validation)
def det_preprocess_batch(batch, training=False):
    """
    Standardize the preprocessing for evaluation.
    Args:
        batch: (B, C, H, W) tensor
    """
    # Simple normalization to [0,1] and then ImageNet stats
    # No heavy augmentations or TTA here for clean visualization
    return batch

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_gradcam(backbone, input_batch, target_layer):
    """
    Generates Grad-CAM heatmap for the ConvNeXt/ResNet backbone.
    """
    backbone.eval()
    activations = []
    gradients = []
    
    def save_activation(module, input, output):
        activations.append(output)
    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    handle_a = target_layer.register_forward_hook(save_activation)
    handle_g = target_layer.register_full_backward_hook(save_gradient)
    
    # Forward pass on backbone
    features = backbone(input_batch)
    # Global average pool (mimicking the end of the backbone or the part being monitored)
    # For ConvNeXt, features are already spatially formatted for pooling
    
    # Here we assume class 1 (Positive) is the target
    # We need a dummy classifier output to calculate gradients if we're only looking at the backbone.
    # Alternatively, we just use the feature volume's maximum activation.
    
    # A cleaner way: use a small head to get "Positive" score
    # Since we can't easily hook the MIL head here, we'll monitor the feature map's sum
    score = features.mean() 
    
    backbone.zero_grad()
    score.backward(retain_graph=False)
    
    handle_a.remove()
    handle_g.remove()
    
    if not gradients or not activations:
        return np.zeros((448, 448)), 0.0

    weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations[0], dim=1, keepdim=True)
    cam = F.relu(cam) # ReLU to only keep activations that positively impact the score
    
    # Normalize heatmap
    cam = cam.detach().cpu().numpy()
    cam = cam - np.min(cam)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    
    # Resize to input size
    from scipy.ndimage import zoom
    h, w = input_batch.shape[2], input_batch.shape[3]
    heatmap = zoom(cam[0, 0], (h / cam.shape[2], w / cam.shape[3]))
    
    return heatmap

def full_visual_report(RUN_ID, MODEL_PATH, MODEL_NAME="convnext_tiny", fold_idx=0, num_folds=5):
    print(f"--- Generating Visual Report for {RUN_ID} (Model: {MODEL_NAME}, Fold: {fold_idx}) ---")
    OUTPUT_DIR = os.path.join("results", f"{RUN_ID}_gradcam_samples")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize Dataset (Hold-out / Validation subset)
    full_dataset = HPyloriDataset(TRAIN_DIR, PATIENT_CSV, PATCH_CSV, transform=VAL_TRANSFORM, train=False)
    
    # Replicate Patient Split
    unique_patients = sorted(list(set([os.path.basename(os.path.dirname(p)) for p, _ in full_dataset.samples])))
    n_total = len(unique_patients)
    fold_size = n_total // num_folds
    val_patients = unique_patients[fold_idx * fold_size : (fold_idx + 1) * fold_size]
    
    val_indices = [i for i, (p, _) in enumerate(full_dataset.samples) 
                  if os.path.basename(os.path.dirname(p)) in val_patients]
    val_dataset = Subset(full_dataset, val_indices)
    
    # Load Model (Attention-MIL Architecture)
    model = get_model(model_name=MODEL_NAME, num_classes=2).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Handle both checkpoint formats (entire state_dict or dict wrapping it)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Remove 'module.' prefix if it exists (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()

    # Determine Target Layer
    if "convnext" in MODEL_NAME:
        target_layer = model.backbone.features[7][2].block[5]
    else:
        target_layer = model.backbone.layer4[2]

    # --- Step 1: Run Inference and Collect Patient-Level Results ---
    all_probs = []
    all_labels = []
    patient_ids = []
    
    # Group indices by patient
    pat_to_indices = {}
    for i in val_indices:
        p_id = os.path.basename(os.path.dirname(full_dataset.samples[i][0]))
        if p_id not in pat_to_indices:
            pat_to_indices[p_id] = []
        pat_to_indices[p_id].append(i)

    print(f"Running Inference on {len(pat_to_indices)} Validation Patients...")
    vram_bag_limit = 500
    
    patient_performance = []

    for p_id, indices in tqdm(pat_to_indices.items()):
        # Load all patches for this patient
        p_subset = Subset(full_dataset, indices)
        p_loader = DataLoader(p_subset, batch_size=vram_bag_limit, shuffle=False)
        
        all_pat_logits = []
        all_pat_attns = []
        
        with torch.no_grad():
            for batch_data in p_loader:
                # Handle both (bag, label) and (bag, label, id) formats
                if len(batch_data) == 3:
                    bag_imgs, labels, _ = batch_data
                else:
                    bag_imgs, labels = batch_data
                    
                bag_imgs = bag_imgs.to(DEVICE)
                logits, attn = model.forward_bag(bag_imgs)
                all_pat_logits.append(logits)
                all_pat_attns.append(attn.cpu())
                label = labels[0].item()

        # Combine MIL Results
        # For AttentionMIL, the forward_bag likely does the aggregation already
        # if it returns a single logit per bag. If it returns per-patch, we aggregate.
        # Assuming our model.py forward_bag returns the patient-level logit:
        final_logits = torch.mean(torch.stack(all_pat_logits), dim=0)
        prob = torch.softmax(final_logits, dim=1)[0, 1].item()
        
        all_probs.append(prob)
        all_labels.append(label)
        patient_ids.append(p_id)
        
        patient_performance.append({
            "Patient": p_id,
            "Label": label,
            "Prob": prob,
            "Pred": 1 if prob >= 0.5 else 0
        })

    perf_df = pd.DataFrame(patient_performance)
    
    # Ensure binary labels for metrics (remap -1 to 1 if generic positive)
    all_labels_bin = [1 if l != 0 else 0 for l in all_labels]
    
    # --- Step 2: Visualization Plots ---
    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels_bin, [1 if p >= 0.5 else 0 for p in all_probs])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"Patient-Level Confusion Matrix (Fold {fold_idx})")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Neg', 'Pos'])
    plt.yticks(tick_marks, ['Neg', 'Pos'])
    
    # Labeling the matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join("results", f"{RUN_ID}_confusion_matrix.png"))
    plt.close()

    # 2. Patient-Level ROC
    fpr, tpr, _ = roc_curve(all_labels_bin, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.title(f"Patient ROC - Fold {fold_idx}")
    plt.legend()
    plt.savefig(os.path.join("results", f"{RUN_ID}_roc_curve.png"))
    plt.close()

    # --- Step 3: Grad-CAM for Top Suspicious Patients ---
    # Pick Top 3 Positives and Top 3 False Negatives (if any)
    top_positives = perf_df[perf_df['Label'] == 1].sort_values('Prob', ascending=False).head(3)
    ghosts = perf_df[(perf_df['Label'] == 1) & (perf_df['Prob'] < 0.5)].sort_values('Prob', ascending=False).head(3)
    
    targets = pd.concat([top_positives, ghosts])

    print(f"Generating Grad-CAM for {len(targets)} patients...")
    for _, row in targets.iterrows():
        p_id = row['Patient']
        is_fn = row['Prob'] < 0.5
        
        # Reload images
        indices = pat_to_indices[p_id]
        p_subset = Subset(full_dataset, indices)
        
        # Get attention weights to pick the most important patches
        all_attns = []
        all_imgs = []
        with torch.no_grad():
            for i in range(len(p_subset)):
                item_data = p_subset[i]
                img = item_data[0]
                
                img_t = img.unsqueeze(0).to(DEVICE)
                _, attn = model.forward_bag(img_t)
                all_attns.append(attn.item())
                all_imgs.append(img)
        
        # Pick top 2 patches by attention
        top_indices = np.argsort(all_attns)[-2:]
        
        for rank, idx in enumerate(top_indices):
            patch_t = all_imgs[idx].unsqueeze(0).to(DEVICE)
            
            with torch.enable_grad():
                heatmap = generate_gradcam(model.backbone, patch_t, target_layer)
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            orig = all_imgs[idx].permute(1, 2, 0).numpy()
            # Unnormalize
            orig = orig * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            orig = np.clip(orig, 0, 1)
            plt.imshow(orig)
            plt.title(f"Patch {idx} (Attn: {all_attns[idx]:.4f})")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(orig)
            plt.imshow(heatmap, cmap='jet', alpha=0.5)
            prefix = "FN_" if is_fn else ""
            plt.title(f"{prefix}Grad-CAM (Prob: {row['Prob']:.4f})")
            plt.axis('off')
            
            plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}{p_id}_rank{rank}.png"), bbox_inches='tight')
            plt.close()

    print(f"Visual report finished. Results in results/{RUN_ID}_*")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H. Pylori Visual Generation")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID (e.g., 112_103186)")
    parser.add_argument("--fold", type=int, default=0, help="Fold index")
    parser.add_argument("--model_name", type=str, default="convnext_tiny", choices=["resnet50", "convnext_tiny"])
    args = parser.parse_args()
    
    # Construct model path
    model_path = f"results/{args.run_id}_f{args.fold}_{args.model_name}_model_brain.pth"
    if not os.path.exists(model_path):
        # Alternative naming convention
        model_path = f"results/{args.run_id}_model_brain.pth"

    if os.path.exists(model_path):
        full_visual_report(args.run_id, model_path, args.model_name, args.fold)
    else:
        print(f"Error: Model not found at {model_path}")

    parser = argparse.ArgumentParser(description="H. Pylori Visual Generation")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID (e.g., 62_102498)")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (matching training)")
    parser.add_argument("--num_folds", type=int, default=5, help="Total number of folds")
    parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet50", "convnext_tiny"],
                         help="Backbone architecture")
    args = parser.parse_args()
    
    # Constructing paths from ID
    # Use the f{fold} nomenclature in filenames
    full_id = f"{args.run_id}_f{args.fold}"
    model_path = f"results/{full_id}_model_brain.pth"
    if not os.path.exists(model_path):
        # Fallback to model_name infix if needed
        model_path = f"results/{full_id}_{args.model_name}_model_brain.pth"
        
    if os.path.exists(model_path):
        full_visual_report(full_id, model_path, args.model_name, args.fold, args.num_folds)
    else:
        print(f"Error: Model not found at {model_path}")
