
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from dataset import HPyloriDataset
from model import get_model
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, 
    precision_recall_curve, average_precision_score
)
from scipy.stats import skew, kurtosis
from torchvision import transforms
from normalization import MacenkoNormalizer
from meta_classifier import HPyMetaClassifier
from PIL import Image

# --- Config ---
RUN_ID = "12_101795" # UPDATED for the Macenko Run
MODEL_PATH = f"results/{RUN_ID}_model_brain.pth"
OUTPUT_DIR = f"results/{RUN_ID}_gradcam_samples"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Paths
BASE_PATH = "/import/fhome/vlia/HelicoDataSet"
if not os.path.exists(BASE_PATH):
    BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), "..", "HelicoDataSet"))

PATIENT_CSV = os.path.join(BASE_PATH, "PatientDiagnosis.csv")
PATCH_CSV = os.path.join(BASE_PATH, "HP_WSI-CoordAnnotatedAllPatches.xlsx")
TRAIN_DIR = os.path.join(BASE_PATH, "CrossValidation/Annotated")

# Macenko Setup
REFERENCE_PATCH_PATH = "/import/fhome/vlia/HelicoDataSet/CrossValidation/Annotated/B22-47_0/01653_Aug8.png"
normalizer = MacenkoNormalizer()
if os.path.exists(REFERENCE_PATCH_PATH):
    print(f"Fitting Macenko Normalizer to reference: {REFERENCE_PATCH_PATH}")
    ref_img = Image.open(REFERENCE_PATCH_PATH).convert("RGB")
    normalizer.fit(ref_img)

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((448, 448)),
    normalizer,
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_gradcam(model, input_batch, target_layer):
    model.eval()
    activations = []
    gradients = []
    
    def save_activation(module, input, output):
        activations.append(output)
    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    handle_a = target_layer.register_forward_hook(save_activation)
    handle_g = target_layer.register_full_backward_hook(save_gradient)
    
    logits = model(input_batch)
    probs = F.softmax(logits, dim=1)
    
    score = logits[:, logits.argmax(dim=1)]
    model.zero_grad()
    score.backward(torch.ones_like(score))
    
    handle_a.remove()
    handle_g.remove()
    
    weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations[0], dim=1, keepdim=True)
    cam = F.relu(cam)
    
    return cam, probs

def full_visual_report(RUN_ID, MODEL_PATH, MODEL_NAME="resnet50", fold_idx=4, num_folds=5):
    print(f"--- Generating Visual Report for {RUN_ID} (Model: {MODEL_NAME}, Fold: {fold_idx}) ---")
    OUTPUT_DIR = os.path.join("results", f"{RUN_ID}_gradcam_samples")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_dataset = HPyloriDataset(TRAIN_DIR, PATIENT_CSV, PATCH_CSV, transform=VAL_TRANSFORM)
    
    # Replicate Patient Split (K-Fold strategy from train.py)
    sample_patient_ids = []
    patient_gt = {} 
    for img_path, label, x, y in full_dataset.samples:
        folder_name = os.path.basename(os.path.dirname(img_path))
        patient_id = folder_name.split('_')[0]
        sample_patient_ids.append(patient_id)
        patient_gt[patient_id] = label
    
    unique_patients = sorted(list(set(sample_patient_ids)))
    np.random.seed(42)
    np.random.shuffle(unique_patients)
    
    # Calculate fold boundaries (exactly like train.py)
    fold_size = len(unique_patients) // num_folds
    val_start = fold_idx * fold_size
    val_end = val_start + fold_size if fold_idx < num_folds - 1 else len(unique_patients)
    
    val_pats = set(unique_patients[val_start:val_end])
    val_indices = [i for i, pid in enumerate(sample_patient_ids) if pid in val_pats]
    
    test_subset = Subset(full_dataset, val_indices)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4)
    
    # Model Selection
    model = get_model(model_name=MODEL_NAME, num_classes=2).to(DEVICE)
    # Handle possible torch.compile prefix and loading
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # If the checkpoint has the _orig_mod key, it was saved from a compiled model
    if any(k.startswith('_orig_mod.') for k in checkpoint.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[10:] if k.startswith('_orig_mod.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    # For patient-level aggregation
    patient_probs = {pid: [] for pid in val_pats}
    patient_coords = {pid: [] for pid in val_pats}
    
    gradcam_saved = 0

    print("Running evaluation on Independent Patient Set...")
    for inputs, labels, paths, coords in tqdm(test_loader):
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            outputs = model(inputs)
            probs_batch = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds_batch = outputs.argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds_batch)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs_batch)
        
        # Track for aggregation
        for i, path in enumerate(paths):
            folder_name = os.path.basename(os.path.dirname(path))
            pid = folder_name.split('_')[0]
            if pid in patient_probs:
                patient_probs[pid].append(probs_batch[i])
                patient_coords[pid].append((coords[0][i].item(), coords[1][i].item()))
        
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

        # Spatial Context (Optional but recommended for consistency)
        clustering_score = 0
        coords_np = np.array(patient_coords[pat_id])
        high_conf_idx = np.where(probs > 0.90)[0]
        if len(high_conf_idx) > 1:
            from sklearn.neighbors import NearestNeighbors
            pts = coords_np[high_conf_idx]
            nbrs = NearestNeighbors(n_neighbors=min(5, len(pts))).fit(pts)
            distances, _ = nbrs.kneighbors(pts)
            clustering_score = 1.0 / (np.mean(distances) + 1.0)

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
            "Patch_Count": len(probs),
            "Spatial_Clustering": clustering_score
        })

    consensus_df = pd.DataFrame(consensus_data)
    
    # Use learned meta-model for patient-level aggregation
    meta = HPyMetaClassifier()
    meta_results = meta.predict(consensus_df)
    
    if meta_results is not None:
        meta_preds, _, meta_probs = meta_results
        pat_probs_final = meta_probs
    else:
        # Fallback to Max probability aggregation
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
