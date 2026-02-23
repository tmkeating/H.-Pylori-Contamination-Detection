import os                       # Standard library for file path management
import torch                    # Core library for deep learning
import torch.nn as nn           # Tools for building neural network layers
import torch.optim as optim     # Mathematical tools to "teach" the model
from torch.optim.adam import Adam # The specific algorithm to adjust the brain
from torch.optim import AdamW    # Better for ConvNeXt architectures
import numpy as np               # Numeric library
import pandas as pd              # Data manipulation library
import matplotlib.pyplot as plt  # Drawing/plotting library
import argparse
from torch.utils.data import DataLoader, random_split, Dataset, Subset, WeightedRandomSampler # Tools to manage and split data
from torchvision import transforms # Tools to prep images for the AI
from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()
from torchvision.transforms import v2 # Faster, optimized transforms
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, 
    classification_report, precision_recall_curve, 
    average_precision_score, PrecisionRecallDisplay
) 
from dataset import HPyloriDataset # Our custom code that finds images/labels
from model import get_model        # Our custom code that builds the AI brain
from tqdm import tqdm              # A library that shows a "progress bar"
import re                          # Regexp to handle file numbering
import gc
import torch.nn.functional as F
from normalization import MacenkoNormalizer
from meta_classifier import HPyMetaClassifier

class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Diagnostic Screening.
    Reduces the relative loss for well-classified examples, focusing on 
    the sparse, hard-to-detect bacterial signals.
    Includes Label Smoothing (Optimization 7) to prevent overconfidence.
    """
    def __init__(self, alpha=1, gamma=2, weight=None, smoothing=0.05):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        # Apply Label Smoothing to the Cross Entropy base (Optimization 7)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight, label_smoothing=self.smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

def generate_gradcam(model, input_batch, target_layer):
    """Generates Grad-CAM heatmaps for a batch of images."""
    model.eval()
    
    # Hooks to store activations and gradients
    activations = []
    gradients = []
    
    def save_activation(module, input, output):
        activations.append(output)
    
    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Attach hooks to the target layer
    handle_a = target_layer.register_forward_hook(save_activation)
    handle_g = target_layer.register_full_backward_hook(save_gradient)
    
    # Forward pass
    logits = model(input_batch)
    probs = F.softmax(logits, dim=1)
    
    # Use the class with highest probability as target
    score = logits[:, logits.argmax(dim=1)]
    model.zero_grad()
    score.backward(torch.ones_like(score))
    
    # Remove hooks
    handle_a.remove()
    handle_g.remove()
    
    # Pool gradients across width/height
    weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
    # Weighted sum of activations
    cam = torch.sum(weights * activations[0], dim=1, keepdim=True)
    # ReLU to keep only positive influence
    cam = F.relu(cam)
    
    return cam, probs

def get_next_run_number(results_dir="results", current_slurm_id=None):
    """
    Finds the next available numeric prefix for output files, 
    respecting SLURM Job ID order to prevent race conditions during 
    multi-fold submissions.
    """
    if not os.path.exists(results_dir):
        return 0
    
    files = os.listdir(results_dir)
    # Map JobID to RunID (to find the most recent anchor)
    job_to_run = {}
    
    # 1. Scan results for existing mappings
    for f in files:
        # Matches formats like "62_102498_..."
        match = re.match(r"^(\d+)_(\d+)_", f)
        if match:
            run_id = int(match.group(1))
            job_id = int(match.group(2))
            if job_id not in job_to_run or run_id > job_to_run[job_id]:
                job_to_run[job_id] = run_id
        
        # Also check existing output logs for "Starting Run ID" 
        if f.startswith("output_") and f.endswith(".txt"):
            jid_match = re.search(r"output_(\d+).txt", f)
            if jid_match:
                jid = int(jid_match.group(1))
                try:
                    with open(os.path.join(results_dir, f), 'r') as log:
                        # Scan top of file for the header we printed
                        for _ in range(50):
                            line = log.readline()
                            if not line: break
                            m = re.search(r"Run ID: (\d+)", line)
                            if m:
                                rid = int(m.group(1))
                                if jid not in job_to_run or rid > job_to_run[jid]:
                                    job_to_run[jid] = rid
                                break
                except: pass

    # If not running in SLURM, use standard max+1 logic
    if current_slurm_id is None or not str(current_slurm_id).isdigit():
        return max(job_to_run.values()) + 1 if job_to_run else 0
        
    current_jid = int(current_slurm_id)
    
    # 2. Identify all Job IDs in the queue (from output_*.txt placeholders)
    all_job_ids = set(job_to_run.keys())
    for f in files:
        if f.startswith("output_") and f.endswith(".txt"):
            jid_m = re.search(r"output_(\d+).txt", f)
            if jid_m:
                all_job_ids.add(int(jid_m.group(1)))
    
    all_job_ids.add(current_jid)
    sorted_jids = sorted(list(all_job_ids))
    
    # 3. Find the most recent anchor point (highest Job ID < current that has a Run ID)
    older_assigned = sorted([jid for jid in job_to_run.keys() if jid < current_jid])
    
    if not older_assigned:
        # No older jobs found with labels. We are the beginning of this results directory.
        return sorted_jids.index(current_jid)
    
    last_known_jid = older_assigned[-1]
    last_known_run = job_to_run[last_known_jid]
    
    # 4. Count the number of 'slots' (output logs) between then and now
    rank_offset = 0
    for jid in sorted_jids:
        if jid > last_known_jid and jid <= current_jid:
            rank_offset += 1
            
    return last_known_run + rank_offset

# This helper class allows us to have different transforms for train and validation split
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        img, label = self.subset[index] 
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)

def train_model(fold_idx=0, num_folds=5, model_name="resnet50"):
    # --- Step 0: Prepare output directories ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get the numeric run ID and the SLURM job ID (if it exists)
    slurm_id = os.environ.get("SLURM_JOB_ID", "local")
    run_id = f"{get_next_run_number(results_dir, slurm_id):02d}"
    prefix = f"{run_id}_{slurm_id}_f{fold_idx}_{model_name}" 
    print(f"--- Starting Run ID: {run_id} (Fold: {fold_idx + 1}/{num_folds}, Model: {model_name}, SLURM Job: {slurm_id}) ---")

    # Define versioned file paths
    best_model_path = os.path.join(results_dir, f"{prefix}_model_brain.pth")
    results_csv_path = os.path.join(results_dir, f"{prefix}_evaluation_report.csv")
    cm_path = os.path.join(results_dir, f"{prefix}_confusion_matrix.png")
    roc_path = os.path.join(results_dir, f"{prefix}_roc_curve.png")
    pr_path = os.path.join(results_dir, f"{prefix}_pr_curve.png")
    history_path = os.path.join(results_dir, f"{prefix}_learning_curves.png")
    hist_path = os.path.join(results_dir, f"{prefix}_probability_histogram.png")
    patient_cm_path = os.path.join(results_dir, f"{prefix}_patient_confusion_matrix.png")
    patient_roc_path = os.path.join(results_dir, f"{prefix}_patient_roc_curve.png")
    patient_pr_path = os.path.join(results_dir, f"{prefix}_patient_pr_curve.png")
    gradcam_dir = os.path.join(results_dir, f"{prefix}_gradcam_samples")
    os.makedirs(gradcam_dir, exist_ok=True)

    # --- Step 1: Choose our study device ---
    # Use a Graphics Card (CUDA) if available; otherwise, use the Main Processor (CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Optimization 5D: Precision and Speed tuning for A40
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print("Set float32 matmul precision to 'high' for A40 hardware optimization.")
    
    # --- Step 2: Set the paths to our data ---
    # We prioritize local scratch space for speed, then fallback to network path
    base_data_path = "/tmp/ricse03_h_pylori_data"
    
    if not os.path.exists(base_data_path):
        base_data_path = "/import/fhome/vlia/HelicoDataSet"

    if not os.path.exists(base_data_path):
        # Fallback for local development or different environments
        local_path = "/home/twyla/Documents/Classes/aprenentatgeProfund/Code/HelicoDataSet"
        if os.path.exists(local_path):
            base_data_path = local_path
        else:
            # Last resort: look in the parent directory
            base_data_path = os.path.abspath(os.path.join(os.getcwd(), "..", "HelicoDataSet"))
        
        print(f"Primary data path not found. Using: {base_data_path}")
    else:
        print(f"Using Data Path: {base_data_path}")

    # --- Step 2.5: Initialize Macenko Normalizer on GPU ---
    from PIL import Image
    normalizer = MacenkoNormalizer()
    
    # Use a fixed reference patch for Macenko normalization consistency
    # We prioritize paths relative to our base_data_path for portability
    rel_ref_path = "CrossValidation/Annotated/B22-47_0/01653.png"
    reference_patch_path = os.path.join(base_data_path, rel_ref_path)
    
    if os.path.exists(reference_patch_path):
        print(f"Fitting Macenko Normalizer (GPU-ready) to reference: {reference_patch_path}")
        ref_img = Image.open(reference_patch_path).convert("RGB")
        normalizer.fit(ref_img, device=device)
    else:
        print(f"WARNING: Reference patch {reference_patch_path} not found. Normalization disabled.")

    patient_csv = os.path.join(base_data_path, "PatientDiagnosis.csv")
    patch_csv = os.path.join(base_data_path, "HP_WSI-CoordAnnotatedAllPatches.xlsx") # Use Excel directly
    train_dir = os.path.join(base_data_path, "CrossValidation/Annotated")
    # This folder contains patients that the AI has NEVER seen during training.
    holdout_dir = os.path.join(base_data_path, "HoldOut")

    # --- Step 3: Define "Study Habits" (Transforms) ---
    # CPU-bound: Minimal prep to save worker time
    train_transform = transforms.Compose([
        transforms.Resize((448, 448)), 
        transforms.ToTensor(), 
    ])

    # GPU-bound: Advanced augmentations performed significantly faster on A40
    gpu_augment = v2.Compose([
        v2.RandomHorizontalFlip(), 
        v2.RandomVerticalFlip(),
        v2.RandomRotation(90),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        v2.RandomGrayscale(p=0.1),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    # Validation habits: No random variety here, just high resolution
    val_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    
    # GPU-based normalization (ImageNet stats)
    gpu_normalize = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # --- Step 4: Load and split the data ---
    # Create the full dataset object without a transform initially
    full_dataset = HPyloriDataset(train_dir, patient_csv, patch_csv, transform=None)
    
    # --- PATIENT-LEVEL SPLIT ---
    # To prevent "Data Leakage", we must ensure that all patches from a single 
    # patient are either in Traing OR Validation, but never both.
    sample_patient_ids = []
    for img_path, label, x, y in full_dataset.samples:
        # Extract patient ID from folder name (e.g. "123_Annotated")
        folder_name = os.path.basename(os.path.dirname(img_path))
        patient_id = folder_name.split('_')[0]
        sample_patient_ids.append(patient_id)
    
    unique_patients = sorted(list(set(sample_patient_ids)))
    np.random.seed(42) # Ensure same shuffle across all fold runs
    np.random.shuffle(unique_patients)
    
    # Calculate fold boundaries
    fold_size = len(unique_patients) // num_folds
    val_start = fold_idx * fold_size
    # Ensure the last fold takes any remainder patients
    val_end = val_start + fold_size if fold_idx < num_folds - 1 else len(unique_patients)
    
    val_patients = set(unique_patients[val_start:val_end])
    train_patients = set([p for p in unique_patients if p not in val_patients])
    
    print(f"--- Fold {fold_idx + 1}/{num_folds} Split ---")
    print(f"Total Patients: {len(unique_patients)}")
    print(f"Training Patients: {len(train_patients)}")
    print(f"Validation Patients: {len(val_patients)}")

    train_indices = [i for i, pid in enumerate(sample_patient_ids) if pid in train_patients]
    val_indices = [i for i, pid in enumerate(sample_patient_ids) if pid in val_patients]
    
    print(f"Independent Patient-level split:")
    print(f" - Train: {len(train_patients)} patients, {len(train_indices)} patches")
    print(f" - Val:   {len(unique_patients)-len(train_patients)} patients, {len(val_indices)} patches")
    
    # Re-apply our study habits for each split
    # Training gets random flips, validation stays as is
    train_data = Subset(full_dataset, train_indices)
    val_data = Subset(full_dataset, val_indices)
    
    # We assign the transforms to the original dataset temporarily during retrieval
    # or better, we use a custom class to apply them
    class TransformDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            img, label, path, coords = self.subset[index]
            if self.transform:
                img = self.transform(img)
            # Return index for Hard Mining (Optimization 6)
            return img, label, path, coords, index
        def __len__(self):
            return len(self.subset)

    train_transformed = TransformDataset(train_data, train_transform)
    val_transformed = TransformDataset(val_data, val_transform)

    # --- Step 4.5: Improve Recall for Contaminated Samples ---
    # We calculate the distribution of our training data
    train_labels = [full_dataset.samples[i][1] for i in train_indices]
    neg_count = train_labels.count(0)
    pos_count = train_labels.count(1)
    print(f"Training distribution: Negative={neg_count}, Contaminated={pos_count}")

    # strategy A: Weighted Sampling (Oversampling)
    # This ensures that every batch of 32 images is balanced (approx 16 Neg, 16 Pos)
    class_weights = [1.0/max(1, neg_count), 1.0/max(1, pos_count)]
    # Use torch tensors for Hard Mining (Optimization 6)
    base_weights = torch.FloatTensor([class_weights[t] for t in train_labels])
    current_weights = base_weights.clone()
    sampler = WeightedRandomSampler(current_weights, len(current_weights))

    # DataLoaders optimized for NVIDIA A40 (48GB VRAM)
    # Reduced batch_size for ResNet50 (Iteration 2)
    batch_size = 64
    train_loader = DataLoader(
        train_transformed, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=7, 
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=4 # Increase buffer of ready batches
    )
    val_loader = DataLoader(
        val_transformed, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=7, 
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=4
    )

    # --- Step 5: Build the customized AI brain ---
    model = get_model(model_name=model_name, num_classes=2, pretrained=True).to(device)

    # --- Step 6: Define the Learning Rules ---
    # strategy B: Focal Loss for Sparse Bacteremia Detection
    # Optimized to ignore common histological background and focus on sparse bacteria.
    loss_weights = torch.FloatTensor([1.0, 1.5]).to(device) # High sensitivity push (Optimization 6.1)
    criterion = FocalLoss(gamma=2, weight=loss_weights)

    # --- Optimization 5D: Preprocessing & Model Compilation (Kernel Fusion) ---
    # We define a fused preprocessing function for deterministic operations.
    # Moving augmentations OUT of the compiled block to avoid Dynamo recompilation.
    def det_preprocess_batch(inputs, training=False):
        # normalize_batch is now fully vectorized (Optimization 5D)
        # Includes optional pathological stain jittering (H&E space)
        inputs = normalizer.normalize_batch(inputs, jitter=training)
        inputs = gpu_normalize(inputs)
        return inputs
    
    # Enable torch.compile for Kernel Fusion (Optimization 5D)
    if hasattr(torch, "compile"):
        print(f"Enabling torch.compile for {model_name} Optimization (5D)...")
        # Compile the model with extreme overhead reduction
        model = torch.compile(model, mode="reduce-overhead")
        # Compile the deterministic preprocessing pipeline (Folds Macenko into GPU Ops)
        # We don't compile det_preprocess_batch because the 'training' flag 
        # would cause recompilation or complex graph breaks. 
        # The vectorized ops in normalize_batch are already fast.
    
    # Optimizer Choice:
    # ConvNeXt is highly sensitive to the training recipe and benefits from AdamW.
    if "convnext" in model_name:
        print(f"Using AdamW Optimizer for {model_name} stability...")
        optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
    else:
        # ResNet default (Adam)
        optimizer = Adam(model.parameters(), lr=2e-5, weight_decay=5e-3)
    
    # --- Step 6.2: Learning Rate Scheduler ---
    # Relaxed patience (3) to prevent premature collapse after Epoch 2 (Run 50)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # --- Step 6.5: Hardware Optimization (Optional) ---
    # Set this to True ONLY if you have an Intel CPU and compatible IPEX installed.
    # Currently set to False to ensure compatibility across all systems.
    USE_IPEX = False 
    
    if USE_IPEX:
        try:
            # We import here so it only tries to load if the user turned it on
            import intel_extension_for_pytorch as ipex # type: ignore
            model, optimizer = ipex.optimize(model, optimizer=optimizer)
            print("Intel Extension for PyTorch optimization enabled.")
        except Exception as e:
            print(f"IPEX failed to load: {e}. Falling back to standard PyTorch.")
    else:
        print("Running on standard PyTorch (IPEX disabled).")

    # --- Step 7: The Main Training Loop ---
    # We use Automatic Mixed Precision (AMP) to speed up training on the A40
    scaler = torch.amp.GradScaler('cuda')
    num_epochs = 15 # Increased for high-specificity convergence (Run 42 pivot)
    best_loss = float('inf')
    
    # Track the "History" to plot learning curves later
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    # Initialize Hard Mining (Optimization 6)
    per_sample_loss = torch.zeros(len(train_indices), device='cpu')
    num_hard_samples = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # --- Study Mode (Train) ---
        model.train() # Tell the brain it is in learning mode
        running_loss = 0.0
        running_corrects = 0
        
        # Hand batches of images to the AI one by one
        # Added indices for Hard Mining (Optimization 6)
        for inputs, labels, paths, coords, indices in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # --- GPU-Based Preprocessing Pipeline (Optimization 5D) ---
            # 1. Apply stochastic augmentations (Outside compilation)
            inputs = gpu_augment(inputs)
            # 2. Apply deterministic normalization with jitter (Training=True)
            inputs = det_preprocess_batch(inputs, training=True)
            
            optimizer.zero_grad()
            
            # Use autocast for the forward pass (Mixed Precision)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                # For OHNM, we need individual losses (no reduction)
                # Apply same label smoothing for consistency (Optimization 7)
                batch_loss_individual = F.cross_entropy(outputs, labels, reduction='none', label_smoothing=0.05)
                loss = criterion(outputs, labels) # Re-compute with weight/focal reduction for optimization
            
            # Store individual losses for sampling weight adjustment (Mining)
            per_sample_loss[indices] = batch_loss_individual.detach().cpu()
            
            _, preds = torch.max(outputs, 1)
            
            # Use scaler for the backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Keep track of scores
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            
        epoch_loss = running_loss / len(train_indices)
        epoch_acc = float(running_corrects) / len(train_indices)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- Volatile Top-10% Stratified Mining (Optimization 6.2) ---
        # We reset weights every epoch to prevent exponential runaway and lock focus
        # to the 10% hardest samples per class.
        current_weights = base_weights.clone()
        train_labels_tensor = torch.LongTensor(train_labels)
        
        # Segment by class to ensure stratified mining
        pos_indices = torch.where(train_labels_tensor == 1)[0]
        neg_indices = torch.where(train_labels_tensor == 0)[0]
        
        # Select top 10% hardest from each class
        k_pos = max(1, int(0.10 * len(pos_indices)))
        k_neg = max(1, int(0.10 * len(neg_indices)))
        
        # Get losses for these groups
        pos_losses = per_sample_loss[pos_indices]
        neg_losses = per_sample_loss[neg_indices]
        
        # Identify hard indices (Top-K)
        _, top_pos_idx = torch.topk(pos_losses, k_pos)
        _, top_neg_idx = torch.topk(neg_losses, k_neg)
        
        hard_pos_indices = pos_indices[top_pos_idx]
        hard_neg_indices = neg_indices[top_neg_idx]
        
        print(f"Hard Mining (Volatile): Boosting top 10% ({len(hard_pos_indices)} Pos, {len(hard_neg_indices)} Neg) samples.")
        
        # Apply multipliers to fresh weights (Iteration 8 logic)
        current_weights[hard_pos_indices] *= 1.5 # Priority boost for bacteria
        current_weights[hard_neg_indices] *= 1.2 # Standard boost for artifacts
        
        # Update the sampler weights in-place
        sampler.weights = current_weights
        
        # --- Self-Test Mode (Validation) ---
        model.eval() # Tell brain it's testing time (no more updating connections)
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad(): # Don't take any math notes, just grade
            # Unpack the 5th index but discard it (Validation)
            for inputs, labels, paths, coords, _ in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # --- GPU-Based Preprocessing Pipeline (Optimization 5D) ---
                # No jitter during validation (Training=False)
                inputs = det_preprocess_batch(inputs, training=False)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                
        val_epoch_loss = val_loss / len(val_indices)
        val_epoch_acc = float(val_corrects) / len(val_indices)
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # --- Step 7.2: Reduce LR on Plateau ---
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_epoch_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")

        # Store history
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)

        # --- Report Card: Save the best version ---
        # If this epoch had the lowest loss yet (best performance considering weights), save it
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} (Val Loss: {val_epoch_loss:.4f})")

    print(f"Training complete. Best Val Loss: {best_loss:.4f}")

    # --- Step 7.3: Extract Difficulty Report (Mining) ---
    # Save the top 100 most difficult patches from the training set
    # These are the ones where the AI learned the most (highest loss)
    top_k = 100
    if len(per_sample_loss) > top_k:
        top_vals, top_idx = torch.topk(per_sample_loss, k=top_k)
        difficulty_data = []
        for i in range(top_k):
            # Map index back to full dataset to get metadata
            global_idx = train_indices[top_idx[i].item()]
            img_path, label, x, y = full_dataset.samples[global_idx]
            difficulty_data.append({
                "Rank": i + 1,
                "Loss": top_vals[i].item(),
                "Path": img_path,
                "Label": "Contaminated" if label == 1 else "Negative",
                "X": x,
                "Y": y
            })
        
        difficulty_df = pd.DataFrame(difficulty_data)
        difficulty_csv_path = os.path.join(results_dir, f"{prefix}_hardest_patches.csv")
        difficulty_df.to_csv(difficulty_csv_path, index=False)
        print(f"Difficulty Report (Top {top_k} Hardest Patches) saved to {difficulty_csv_path}")

    # --- Step 7.4: Memory Cleanup ---
    # Delete training loaders and clear cache to free up RAM for the Hold-Out test
    print(f"Fold {fold_idx} finished. Max Hard Samples: {num_hard_samples}")
    del train_loader
    del val_loader
    del train_transformed
    del val_transformed
    del full_dataset # Reclaim dataset memory
    del train_data
    del val_data
    del num_hard_samples
    gc.collect()
    torch.cuda.empty_cache()

    # --- Step 7.5: Save Learning Curves ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.savefig(history_path)
    print(f"Saved learning curves to {history_path}")

    # --- Step 8: Final Evaluation (Independent Patients) ---
    # This set contains patients that the AI has NEVER seen during training.
    # This is the "Gold Standard" test for avoiding Data Leakage.
    print(f"\nEvaluating on Independent Hold-Out set from: {holdout_dir}")
    
    # Load the TRULY independent data
    holdout_dataset = HPyloriDataset(holdout_dir, patient_csv, patch_csv, transform=None)
    holdout_transformed = TransformDataset(Subset(holdout_dataset, range(len(holdout_dataset))), val_transform)
    
    # Reduced workers to 0 to eliminate worker process memory overhead in SLURM (Run 61 final Fix)
    holdout_loader = DataLoader(
        holdout_transformed, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, # Most stable for high-memory environments
        pin_memory=True
    )
    
    # Efficient calculation of patient count
    pids_holdout = [os.path.basename(os.path.dirname(p)).split('_')[0] for p, l, x, y in holdout_dataset.samples]
    print(f"Independent Patients in Hold-Out: {len(set(pids_holdout))}")
    print(f"Total Patches in Hold-Out: {len(holdout_dataset)}")
    del pids_holdout # Reclaim string list memory

    # Load our best saved brain
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    
    # Pre-allocate numpy arrays for maximum memory efficiency (Run 62 Fix)
    num_holdout = len(holdout_dataset)
    all_preds = np.zeros(num_holdout, dtype=np.int8)
    all_labels = np.zeros(num_holdout, dtype=np.int8)
    all_probs = np.zeros(num_holdout, dtype=np.float32)
    all_coords = np.zeros((num_holdout, 2), dtype=np.int32)
    gradcam_samples = []

    pointer = 0
    with torch.no_grad():
        # Unpack indices (discarded for test set evaluation)
        for inputs, labels, paths, coords, _ in tqdm(holdout_loader, desc="Patient-Independent Test"):
            batch_size_actual = inputs.size(0)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # No jitter during test (Training=False)
            inputs = det_preprocess_batch(inputs, training=False)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            thresh = 0.2
            preds = (probs[:, 1] > thresh).long()
            
            # Fill pre-allocated arrays
            all_preds[pointer:pointer+batch_size_actual] = preds.cpu().numpy()
            all_labels[pointer:pointer+batch_size_actual] = labels.cpu().numpy()
            all_probs[pointer:pointer+batch_size_actual] = probs[:, 1].cpu().numpy()
            all_coords[pointer:pointer+batch_size_actual] = coords.cpu().numpy()
            
            if len(gradcam_samples) < 10:
                pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
                if len(pos_indices) > 0:
                    idx = pos_indices[0].item()
                    gradcam_samples.append(('Real_Pos', inputs[idx:idx+1].clone()))
                
                fp_indices = ((labels == 0) & (probs[:, 1] > 0.9)).nonzero(as_tuple=True)[0]
                if len(fp_indices) > 0:
                    idx = fp_indices[0].item()
                    gradcam_samples.append(('False_Alarm_Artifact', inputs[idx:idx+1].clone()))
            
            pointer += batch_size_actual
            
            # Periodically force cleanup
            if pointer % (batch_size * 50) == 0:
                torch.cuda.empty_cache()
                gc.collect()

    gc.collect()

    # --- Step 9: Detailed Reporting ---
    
    # 1. Classification Report (Precision, Recall, F1)
    print("\nClassification Report:")
    report_dict = classification_report(all_labels, all_preds, target_names=['Negative', 'Contaminated'], output_dict=True, zero_division=0)
    print(classification_report(all_labels, all_preds, target_names=['Negative', 'Contaminated'], zero_division=0))

    # 2. Save machine-readable CSV for AI evaluation
    results_df = pd.DataFrame(report_dict).transpose()
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Add overall summary stats
    results_df.loc['OVERALL_AUC_ROC', 'support'] = roc_auc 
    results_df.to_csv(results_csv_path)
    print(f"Saved machine-readable report to {results_csv_path}")

    # 3. Save Confusion Matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Contaminated'])
    disp.plot(cmap='Blues') # Use the string name to avoid linter confusion
    plt.title("HoldOut Set: Confusion Matrix")
    plt.savefig(cm_path)
    print(f"Saved {cm_path}")

    # 4. Save ROC Curve plot
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(roc_path)
    print(f"Saved {roc_path}")

    # 5. Save PR Curve plot
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)
    ap_score = average_precision_score(all_labels, all_probs)
    plt.figure()
    plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (AP = {ap_score:0.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(pr_path)
    print(f"Saved {pr_path}")

    # 6. Save Probability Histograms
    plt.figure()
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    plt.hist(all_probs[all_labels == 1], bins=20, alpha=0.5, label='Actual Positive', color='red')
    plt.hist(all_probs[all_labels == 0], bins=20, alpha=0.5, label='Actual Negative', color='blue')
    plt.xlabel('Probability of "Contaminated"')
    plt.ylabel('Number of Samples')
    plt.title('Predicted Probability Distribution')
    plt.legend()
    plt.savefig(hist_path)
    print(f"Saved {hist_path}")

    # 7. Generate Grad-CAM for saved samples
    if gradcam_samples:
        print(f"Generating Grad-CAM interpretability maps for {model_name}...")
        actual_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        
        # Architecture-aware target layer selection
        if "resnet50" in model_name:
            target_layer = actual_model.backbone.layer4[-1]
        elif "convnext" in model_name:
            # For ConvNeXt, the last feature block contains the final spatial activations
            target_layer = actual_model.backbone.features[-1]
        else:
            target_layer = None

        if target_layer:
            for i, (sample_type, img_tensor) in enumerate(gradcam_samples):
                with torch.enable_grad(): # Grad-CAM needs gradients
                    # Use the uncompiled model (actual_model) for Grad-CAM to ensure hooks trigger
                    cam, prob = generate_gradcam(actual_model, img_tensor, target_layer)
                
                # Prepare image and CAM for display
                img = img_tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy()
                # Un-normalize for display
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                cam = cam.squeeze().detach().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.title(f"{sample_type} (Prob: {prob[0,1]:.2f})")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(img)
                plt.imshow(cam, cmap='jet', alpha=0.5)
                plt.title("Grad-CAM Heatmap")
                plt.axis('off')
                
                plt.savefig(os.path.join(gradcam_dir, f"{sample_type}_{i}.png"))
                plt.close()
            print(f"Saved Grad-CAM samples to {gradcam_dir}")

    # --- Step 10: Patient-Level Consensus Analysis ---
    # In the real world, a doctor doesn't care about one patch; they care if the PATIENT has H. Pylori.
    print("\n--- Patient-Level Consensus Report (Independent Set) ---")
    patient_probs = {} # { patientID: [prob1, prob2, ...] }
    patient_coords = {} # { patientID: [coord1, coord2, ...] }
    patient_gt = {}    # { patientID: label }
    
    for idx, prob in enumerate(all_probs):
        img_path, _, _, _ = holdout_dataset.samples[idx]
        label = all_labels[idx]
        coord = all_coords[idx]
        
        folder_name = os.path.basename(os.path.dirname(img_path))
        pat_id = folder_name.split('_')[0]
        
        if pat_id not in patient_probs:
            patient_probs[pat_id] = []
            patient_coords[pat_id] = []
            patient_gt[pat_id] = label
        patient_probs[pat_id].append(prob)
        patient_coords[pat_id].append(coord)
    
    consensus_data = []
    for pat_id, probs in patient_probs.items():
        probs_np = np.array(probs)
        coords_np = np.array(patient_coords[pat_id])
        
        avg_prob = np.mean(probs_np)
        max_prob = np.max(probs_np)
        min_prob = np.min(probs_np)
        std_prob = np.std(probs_np)
        med_prob = np.median(probs_np)
        p10, p25, p75, p90 = np.percentile(probs_np, [10, 25, 75, 90])
        
        prob_series = pd.Series(probs_np)
        skew = prob_series.skew() if len(probs_np) > 2 else 0
        kurt = prob_series.kurt() if len(probs_np) > 3 else 0
        
        # --- Spatial Clustering Logic (New for Run 58) ---
        # Calculate how 'clumped' the high-probability patches are.
        # High confidence patches (>0.7) that are close to each other indicate biological colonies.
        high_conf_idx = np.where(probs_np > 0.70)[0]
        clustering_score = 0
        if len(high_conf_idx) > 1:
            from sklearn.neighbors import NearestNeighbors
            # Use X, Y coordinates to find nearest neighbors among hot patches
            pts = coords_np[high_conf_idx]
            if len(pts) > 1:
                nbrs = NearestNeighbors(n_neighbors=min(5, len(pts))).fit(pts)
                distances, _ = nbrs.kneighbors(pts)
                # Lower average distance means higher clustering
                clustering_score = 1.0 / (np.mean(distances) + 1.0)
        
        high_conf_count = sum(1 for p in probs if p > 0.90)
        is_dense = high_conf_count >= 40
        is_consistent = (avg_prob > 0.88 and (max_prob - avg_prob) < 0.20 and len(probs) >= 10)
        
        if is_dense or is_consistent:
            pred_label = 1
        else:
            pred_label = 0
            
        actual_label = patient_gt[pat_id]
        
        consensus_data.append({
            "PatientID": pat_id,
            "Actual": 1 if actual_label == 1 else 0,
            "Predicted": pred_label,
            "Mean_Prob": avg_prob,
            "Max_Prob": max_prob,
            "Min_Prob": min_prob,
            "Std_Prob": std_prob,
            "Median_Prob": med_prob,
            "P10_Prob": p10,
            "P25_Prob": p25,
            "P75_Prob": p75,
            "P90_Prob": p90,
            "Skew": skew,
            "Kurtosis": kurt,
            "Count_P50": sum(1 for p in probs if p > 0.50),
            "Count_P60": sum(1 for p in probs if p > 0.60),
            "Count_P70": sum(1 for p in probs if p > 0.70),
            "Count_P80": sum(1 for p in probs if p > 0.80),
            "Count_P90": high_conf_count,
            "Patch_Count": len(probs),
            "Spatial_Clustering": clustering_score,
            "Correct": 1 if pred_label == actual_label else 0
        })
    
    consensus_df = pd.DataFrame(consensus_data)
    
    # Try using learned Meta-Classifier (Random Forest)
    meta = HPyMetaClassifier()
    meta_results = meta.predict(consensus_df)
    
    if meta_results is not None:
        meta_preds, meta_reliability = meta_results
        print("Using Learned Meta-Classifier for Diagnosis...")
        consensus_df["Predicted"] = meta_preds
        consensus_df["Confidence"] = meta_reliability
        consensus_df["Method"] = "RandomForest"
    else:
        print("No Meta-Classifier found. Using Heuristic Gates (Fallback)...")
        consensus_df["Method"] = "HeuristicGate"
        consensus_df["Confidence"] = 1.0 # Heuristic is binary/hard
        
    # Recalculate 'Correct' after potential meta-prediction update
    consensus_df["Correct"] = (consensus_df["Predicted"] == consensus_df["Actual"]).astype(int)
    
    consensus_report_path = os.path.join(results_dir, f"{prefix}_patient_consensus.csv")
    consensus_df.to_csv(consensus_report_path, index=False)
    print(f"Patient Consensus Report saved to: {consensus_report_path}")
    
    # Print summary for visibility
    # Ensure all patients are shown in the log
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # Create a pretty-print version with words instead of numbers
    print_df = consensus_df.copy()
    print_df["Actual"] = print_df["Actual"].map({1: "Positive", 0: "Negative"})
    print_df["Predicted"] = print_df["Predicted"].map({1: "Positive", 0: "Negative"})
    print_df["Correct"] = print_df["Correct"].map({1: "Yes", 0: "No"})
    
    print("\nDetailed Patient-Level Metrics (Clinical Report):")
    print(print_df[[
        "PatientID", "Actual", "Predicted", "Mean_Prob", "Max_Prob", 
        "Count_P90", "Patch_Count", "Spatial_Clustering", "Correct"
    ]].to_string(index=False))

    # --- Step 11: Patient-Level Visualizations ---
    print("\nGenerating Patient-Level Visualizations...")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm_pat = confusion_matrix(consensus_df["Actual"], consensus_df["Predicted"])
    disp_pat = ConfusionMatrixDisplay(confusion_matrix=cm_pat, display_labels=['Negative', 'Positive'])
    disp_pat.plot(cmap='Blues', values_format='d')
    plt.title(f"Patient-Level Confusion Matrix (HoldOut)")
    plt.savefig(patient_cm_path)
    plt.close()

    # 2. ROC Curve (using Max_Prob as the continuous score)
    fpr_pat, tpr_pat, _ = roc_curve(consensus_df["Actual"], consensus_df["Max_Prob"])
    roc_auc_pat = auc(fpr_pat, tpr_pat)
    plt.figure()
    plt.plot(fpr_pat, tpr_pat, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_pat:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Patient-Level ROC (Score: Max Patch Prob)')
    plt.legend(loc="lower right")
    plt.savefig(patient_roc_path)
    plt.close()

    # 3. Precision-Recall Curve
    prec_pat, rec_pat, _ = precision_recall_curve(consensus_df["Actual"], consensus_df["Max_Prob"])
    ap_pat = average_precision_score(consensus_df["Actual"], consensus_df["Max_Prob"])
    plt.figure()
    plt.plot(rec_pat, prec_pat, color='blue', lw=2, label=f'PR curve (AP = {ap_pat:0.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Patient-Level Precision-Recall')
    plt.legend(loc="lower left")
    plt.savefig(patient_pr_path)
    plt.close()
    
    print(f"Saved patient-level plots to {results_dir}")

    # Calculate Patient-Level Accuracy
    pat_acc = consensus_df["Correct"].mean() * 100
    print(f"\nFinal Patient-Level Accuracy: {pat_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H. Pylori K-Fold Training")
    parser.add_argument("--fold", type=int, default=0, help="Index of the fold to use for validation (0 to num_folds-1)")
    parser.add_argument("--num_folds", type=int, default=5, help="Total number of folds")
    parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet50", "convnext_tiny"], 
                        help="Backbone architecture to use (resnet50/convnext_tiny)")
    args = parser.parse_args()
    
    train_model(fold_idx=args.fold, num_folds=args.num_folds, model_name=args.model_name)
