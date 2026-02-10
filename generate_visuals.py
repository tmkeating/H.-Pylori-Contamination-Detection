
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

from torchvision import transforms

# --- Config ---
RUN_ID = "08_101766"
MODEL_PATH = f"results/{RUN_ID}_model_brain.pth"
OUTPUT_DIR = f"results/{RUN_ID}_gradcam_samples"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Paths (Matching train.py exactly)
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

def post_hoc_visuals():
    print(f"Loading dataset and model {MODEL_PATH}...")
    # Initialize with proper paths and transform
    full_dataset = HPyloriDataset(TRAIN_DIR, PATIENT_CSV, PATCH_CSV, transform=VAL_TRANSFORM)
    
    # Replicate the HoldOut split logic from train.py (80/20 split of Annotated)
    indices = list(range(len(full_dataset)))
    train_size = int(0.8 * len(full_dataset))
    
    np.random.seed(42)
    np.random.shuffle(indices)
    val_indices = indices[train_size:]
    test_subset = Subset(full_dataset, val_indices)
    
    model = get_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    target_layer = model.layer4[-1]
    
    count = 0
    max_samples = 10
    
    # Use DataLoader to avoid linter type-inference issues with Subset indexing
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
    
    print("Searching for contaminated samples in HoldOut...")
    for img_tensor, label in tqdm(test_loader):
        if label.item() == 1: # Contaminated
            img_batch = img_tensor.to(DEVICE)
            
            with torch.enable_grad():
                cam, prob = generate_gradcam(model, img_batch, target_layer)
            
            # Prepare image for display (undo normalization)
            # img_tensor is [1, 3, 448, 448] from DataLoader
            img = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            cam_np = cam.squeeze().detach().cpu().numpy()
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f"Original (Prob: {prob[0,1]:.2f})")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(img)
            plt.imshow(cam_np, cmap='jet', alpha=0.5)
            plt.title("Grad-CAM Heatmap")
            plt.axis('off')
            
            plt.savefig(f"{OUTPUT_DIR}/sample_{count}.png")
            plt.close()
            
            count += 1
            if count >= max_samples:
                break
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f"Original (Prob: {prob[0,1]:.2f})")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(img)
            plt.imshow(cam_np, cmap='jet', alpha=0.5)
            plt.title("Grad-CAM Heatmap")
            plt.axis('off')
            
            plt.savefig(f"{OUTPUT_DIR}/sample_{count}.png")
            plt.close()
            
            count += 1
            if count >= max_samples:
                break
                
    print(f"Successfully generated {count} Grad-CAM samples in {OUTPUT_DIR}")

if __name__ == "__main__":
    post_hoc_visuals()
