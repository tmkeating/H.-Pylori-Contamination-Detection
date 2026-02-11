
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from dataset import HPyloriDataset
from model import get_model
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, 
    precision_recall_curve, average_precision_score
)
from torchvision import transforms

# --- Config ---
RUN_ID = "10_101782"
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

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((448, 448)),
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

def full_visual_report():
    print(f"--- Generating Visual Report for {RUN_ID} ---")
    full_dataset = HPyloriDataset(TRAIN_DIR, PATIENT_CSV, PATCH_CSV, transform=VAL_TRANSFORM)
    
    # Replicate Patient Split
    sample_patient_ids = []
    for img_path, label in full_dataset.samples:
        folder_name = os.path.basename(os.path.dirname(img_path))
        patient_id = folder_name.split('_')[0]
        sample_patient_ids.append(patient_id)
    
    unique_patients = sorted(list(set(sample_patient_ids)))
    np.random.seed(42)
    np.random.shuffle(unique_patients)
    num_train_pats = int(0.8 * len(unique_patients))
    val_pats = set(unique_patients[num_train_pats:])
    val_indices = [i for i, pid in enumerate(sample_patient_ids) if pid in val_pats]
    
    test_subset = Subset(full_dataset, val_indices)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4)
    
    model = get_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    gradcam_saved = 0

    print("Running evaluation on Independent Patient Set...")
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy()[:, 1])
        
        # Save a few Grad-CAMs while we are at it
        if gradcam_saved < 10:
            pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
            for idx in pos_indices:
                if gradcam_saved >= 10: break
                img_batch = inputs[idx:idx+1]
                target_layer = model.layer4[-1]
                with torch.enable_grad():
                    cam, prob = generate_gradcam(model, img_batch, target_layer)
                
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
                plt.title("Grad-CAM Focus")
                plt.axis('off')
                plt.savefig(f"{OUTPUT_DIR}/sample_{gradcam_saved}.png")
                plt.close()
                gradcam_saved += 1

    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap='Blues')
    plt.title(f"Run {RUN_ID} Confusion Matrix (Patient-Independent)")
    plt.savefig(f"results/{RUN_ID}_confusion_matrix.png")
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f"results/{RUN_ID}_roc_curve.png")
    
    # 3. PR Curve
    prec, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    plt.figure()
    plt.plot(recall, prec, color='blue', label=f'PR (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(f"results/{RUN_ID}_pr_curve.png")

    print(f"Visual report finished. Files saved to results/")

if __name__ == "__main__":
    full_visual_report()
