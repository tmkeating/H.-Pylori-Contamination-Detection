
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
    model.load_state_dict(checkpoint['model_state_dict'])
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
            for bag_imgs, labels in p_loader:
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
    
    # --- Step 2: Visualization Plots ---
    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, [1 if p >= 0.5 else 0 for p in all_probs])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Neg', 'Pos'])
    disp.plot(cmap='Blues')
    plt.title(f"Patient-Level Confusion Matrix (Fold {fold_idx})")
    plt.savefig(os.path.join("results", f"{RUN_ID}_confusion_matrix.png"))
    plt.close()

    # 2. Patient-Level ROC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
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
                img, _ = p_subset[i]
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

        
        # Save a few Grad-CAMs while we are at it
        if gradcam_saved < 10:
            pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
            for idx in pos_indices:
                if gradcam_saved >= 10: break
                img_batch = inputs[idx:idx+1]
                actual_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                
                # Dynamic target layer selection based on architecture
                if "resnet50" in MODEL_NAME:
                    target_layer = actual_model.backbone.layer4[-1]
                elif "convnext" in MODEL_NAME:
                    target_layer = actual_model.backbone.features[-1]
                else:
                    target_layer = None # Fallback
                
                if target_layer:
                    with torch.enable_grad():
                        cam, prob = generate_gradcam(actual_model, img_batch, target_layer)
                    
                    # Plot
                    img = img_batch.squeeze().permute(1, 2, 0).detach().cpu().numpy()
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img = np.clip(img, 0, 1)
                    cam_np = cam.squeeze().detach().cpu().numpy()
                    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
                    
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(img)
                    plt.title(f"Original (Prob Pos: {prob[0,1]:.2f})")
                    plt.axis('off')
                    plt.subplot(1, 2, 2)
                    plt.imshow(img)
                    plt.imshow(cam_np, cmap='jet', alpha=0.5)
                    plt.title(f"Grad-CAM ({MODEL_NAME})")
                    plt.axis('off')
                    plt.savefig(f"{OUTPUT_DIR}/sample_{gradcam_saved}.png")
                    plt.close()
                    gradcam_saved += 1

    # --- Step 1: Patch-Level Plots ---
    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap='Blues')
    plt.title(f"Patch-Level CM (Independent)")
    plt.savefig(f"results/{RUN_ID}_confusion_matrix.png")
    plt.close()
    
    # 2. ROC/PR Curves
    fpr_patch, tpr_patch, _ = roc_curve(all_labels, all_probs)
    roc_auc_patch = auc(fpr_patch, tpr_patch)
    prec_patch, recall_patch, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc_patch = average_precision_score(all_labels, all_probs)

    # --- Step 2: Patient-Level Aggregation (Learned or Heuristic) ---
    print("\nAggregating Patient-Level Results...")
    pat_ids, pat_labels, pat_probs_max, pat_probs_mean = [], [], [], []
    consensus_data = []

    for pat_id in val_pats:
        probs = np.array(patient_probs[pat_id])
        if len(probs) == 0: continue
        
        # Calculate clinical features exactly like train.py
        avg_prob = np.mean(probs)
        max_prob = np.max(probs)
        min_prob = np.min(probs)
        std_prob = np.std(probs)
        med_prob = np.median(probs)
        p10 = np.percentile(probs, 10)
        p25 = np.percentile(probs, 25)
        p75 = np.percentile(probs, 75)
        p90 = np.percentile(probs, 90)
        skew_val = skew(probs) if len(probs) > 2 else 0
        kurt_val = kurtosis(probs) if len(probs) > 2 else 0
        p90_count = sum(1 for p in probs if p > 0.90)

        # Build feature vector
        pat_ids.append(pat_id)
        pat_labels.append(patient_gt[pat_id])
        pat_probs_max.append(max_prob)
        pat_probs_mean.append(avg_prob)
        
        consensus_data.append({
            "PatientID": pat_id,
            "Actual": 1 if patient_gt[pat_id] == 1 else 0,
            "Mean_Prob": avg_prob,
            "Max_Prob": max_prob,
            "Min_Prob": min_prob,
            "Std_Prob": std_prob,
            "Median_Prob": med_prob,
            "P10_Prob": p10,
            "P25_Prob": p25,
            "P75_Prob": p75,
            "P90_Prob": p90,
            "Skew": skew_val,
            "Kurtosis": kurt_val,
            "Count_P50": sum(1 for p in probs if p > 0.50),
            "Count_P60": sum(1 for p in probs if p > 0.60),
            "Count_P70": sum(1 for p in probs if p > 0.70),
            "Count_P80": sum(1 for p in probs if p > 0.80),
            "Count_P90": p90_count,
            "Patch_Count": len(probs)
        })

    consensus_df = pd.DataFrame(consensus_data)
    
    # Fallback to Max probability aggregation (Meta-classifier deprecated in Iteration 10)
    pat_probs_final = pat_probs_max

    # --- Step 3: Patient-Level Plots ---
    # 1. Patient ROC (Comparing Meta-Classifier, Patch-Level, Max Prob, and Suspicious Count)
    fpr_pat, tpr_pat, _ = roc_curve(pat_labels, pat_probs_final)
    roc_auc_pat = auc(fpr_pat, tpr_pat)
    
    fpr_max, tpr_max, _ = roc_curve(pat_labels, pat_probs_max)
    roc_auc_max = auc(fpr_max, tpr_max)
    
    fpr_susp, tpr_susp, _ = roc_curve(pat_labels, consensus_df["Count_P90"])
    roc_auc_susp = auc(fpr_susp, tpr_susp)
    
    plt.figure()
    plt.plot(fpr_pat, tpr_pat, color='darkorange', lw=3, label=f'Meta-Classifier (AUC = {roc_auc_pat:.4f})')
    plt.plot(fpr_max, tpr_max, color='green', linestyle='--', alpha=0.8, label=f'Max Probability (AUC = {roc_auc_max:.4f})')
    plt.plot(fpr_susp, tpr_susp, color='purple', linestyle=':', alpha=0.8, label=f'Suspicious Count (AUC = {roc_auc_susp:.4f})')
    plt.plot(fpr_patch, tpr_patch, color='black', linestyle='--', alpha=0.3, label=f'Patch-Level (AUC = {roc_auc_patch:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Patient-Level ROC: Clinical Comparison')
    plt.legend(loc="lower right")
    plt.savefig(f"results/{RUN_ID}_patient_roc_curve.png")
    plt.close()
    
    # 2. Patient PR
    prec_pat, recall_pat, _ = precision_recall_curve(pat_labels, pat_probs_final)
    pr_auc_pat = average_precision_score(pat_labels, pat_probs_final)
    
    prec_max, recall_max, _ = precision_recall_curve(pat_labels, pat_probs_max)
    pr_auc_max = average_precision_score(pat_labels, pat_probs_max)
    
    prec_susp, recall_susp, _ = precision_recall_curve(pat_labels, consensus_df["Count_P90"])
    pr_auc_susp = average_precision_score(pat_labels, consensus_df["Count_P90"])
    
    plt.figure()
    plt.plot(recall_pat, prec_pat, color='blue', lw=3, label=f'Meta-Classifier (AP = {pr_auc_pat:.4f})')
    plt.plot(recall_max, prec_max, color='green', linestyle='--', alpha=0.8, label=f'Max Probability (AP = {pr_auc_max:.4f})')
    plt.plot(recall_susp, prec_susp, color='purple', linestyle=':', alpha=0.8, label=f'Suspicious Count (AP = {pr_auc_susp:.4f})')
    plt.plot(recall_patch, prec_patch, color='black', linestyle='--', alpha=0.3, label=f'Patch-Level (AP = {pr_auc_patch:.4f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Patient-Level Precision-Recall: Clinical Comparison')
    plt.legend(loc="lower left")
    plt.savefig(f"results/{RUN_ID}_patient_pr_curve.png")
    plt.close()

    print(f"Visual report finished. Patient Accuracy: {consensus_df['Actual'].count()} samples analyzed.")

if __name__ == "__main__":
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
