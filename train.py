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
    
    # Normalize
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-7)
    
    return cam.detach().cpu().numpy(), probs.detach().cpu().numpy()

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

def train_model(fold_idx=0, num_folds=5, model_name="convnext_tiny"):
    """
    Train a deep learning model for H. pylori contamination detection using k-fold cross-validation.
    This function implements a complete machine learning pipeline including:
    - Patient-level k-fold stratification to prevent data leakage
    - GPU-optimized training with automatic mixed precision (AMP)
    - Focal loss with class weighting for imbalanced bacteremia detection
    - OneCycle learning rate scheduling with warmup and cosine annealing
    - Gradient accumulation for effective batch size scaling
    - Patient-level consensus prediction via Attention-MIL aggregation
    - Comprehensive evaluation metrics: patch-level and patient-level confusion matrices, ROC/PR curves
    - Interpretability via Grad-CAM visualization
    - Hardware optimization (torch.compile kernel fusion, Intel IPEX support)
    Args:
        fold_idx (int, optional): Current fold index for k-fold cross-validation. Default: 0
        num_folds (int, optional): Total number of folds for cross-validation. Default: 5
        model_name (str, optional): Model architecture ('convnext_tiny', 'resnet50', etc.). Default: "convnext_tiny"
    Returns:
        None. Outputs saved to results/ directory with files prefixed by {run_id}_{slurm_id}_f{fold_idx}_{model_name}:
        - *_model_brain.pth: Best model weights (lowest validation loss)
        - *_evaluation_report.csv: Patch-level classification metrics
        - *_confusion_matrix.png: Patch-level confusion matrix
        - *_roc_curve.png: Patch-level ROC curve
        - *_pr_curve.png: Patch-level precision-recall curve
        - *_learning_curves.png: Training/validation loss and accuracy over epochs
        - *_probability_histogram.png: Distribution of predicted probabilities
        - *_patient_confusion_matrix.png: Patient-level confusion matrix
        - *_patient_roc_curve.png: Patient-level ROC curve
        - *_patient_pr_curve.png: Patient-level precision-recall curves
        - *_patient_consensus.csv: Patient-level predictions with consensus metrics
        - *_gradcam_samples/: Interpretability maps for sample patches
    Raises:
        FileNotFoundError: If base data path cannot be located
        RuntimeError: If CUDA is unavailable (gracefully falls back to CPU)
    Notes:
        - Patient-level split ensures all patches from a single patient are in train OR validation, never both
        - Grad-CAM requires torch.enable_grad() context and uses the uncompiled model (_orig_mod) for hook activation
        - Hardware: Optimized for NVIDIA A40 (48GB VRAM) with batch_size=128 and accumulation_steps=2
    """
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
    # rel_ref_path = "CrossValidation/Annotated/B22-47_0/01653.png"
    # reference_patch_path = os.path.join(base_data_path, rel_ref_path)
    
    # if os.path.exists(reference_patch_path):
    #     print(f"Fitting Macenko Normalizer (GPU-ready) to reference: {reference_patch_path}")
    #     ref_img = Image.open(reference_patch_path).convert("RGB")
    #     normalizer.fit(ref_img, device=device)
    # else:
    #     print(f"WARNING: Reference patch {reference_patch_path} not found. Normalization disabled.")
    
    print("Pre-processing: Using Standard ImageNet Normalization (IHC-mode).")

    patient_csv = os.path.join(base_data_path, "PatientDiagnosis.csv")
    patch_csv = os.path.join(base_data_path, "HP_WSI-CoordAnnotatedAllPatches.xlsx") # Use Excel directly
    train_dir = os.path.join(base_data_path, "CrossValidation/Annotated")
    # This folder contains patients that the AI has NEVER seen during training.
    holdout_dir = os.path.join(base_data_path, "HoldOut")

    # --- Step 3: Define "Study Habits" (Transforms) ---
    # CPU-bound: Minimal prep to save worker time
    # Turned off upscaling: patches are original 256x256
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(), 
    ])

    # GPU-bound: Advanced augmentations performed significantly faster on A40
    # IHC Pivot: Increased Jitter to compensate for removing Macenko
    gpu_augment = v2.Compose([
        v2.RandomHorizontalFlip(), 
        v2.RandomVerticalFlip(),
        v2.RandomRotation(90),
        v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
        v2.RandomGrayscale(p=0.15),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    # Validation habits: No random variety here, just high resolution
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # GPU-based normalization (ImageNet stats)
    gpu_normalize = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # --- Step 4: Load and split the data ---
    # Create the full dataset object in bag mode for MIL
    # max_bag_size=500 to keep A40 VRAM consumption manageable
    full_dataset = HPyloriDataset(train_dir, patient_csv, patch_csv, transform=None, bag_mode=True, max_bag_size=500, train=True)
    
    # --- PATIENT-LEVEL SPLIT ---
    # In bag mode, full_dataset.bags is a list of (paths, label, patient_id)
    unique_patients = [bag[2] for bag in full_dataset.bags]
    np.random.seed(42) # Ensure same shuffle across all fold runs
    np.random.shuffle(unique_patients)
    
    # Calculate fold boundaries
    fold_size = len(unique_patients) // num_folds
    val_start = fold_idx * fold_size
    # Ensure the last fold takes any remainder patients
    val_end = val_start + fold_size if fold_idx < num_folds - 1 else len(unique_patients)
    
    val_patients = set(unique_patients[val_start:val_end])
    train_patients = set([p for p in unique_patients if p not in val_patients])
    
    print(f"--- Fold {fold_idx + 1}/{num_folds} Split (MIL Bag Mode) ---")
    print(f"Total Patients (Bags): {len(unique_patients)}")
    print(f"Training Patients: {len(train_patients)}")
    print(f"Validation Patients: {len(val_patients)}")

    train_indices = [i for i, bag in enumerate(full_dataset.bags) if bag[2] in train_patients]
    val_indices = [i for i, bag in enumerate(full_dataset.bags) if bag[2] in val_patients]
    
    print(f"Independent Patient-level split:")
    print(f" - Train: {len(train_patients)} bags")
    print(f" - Val:   {len(val_patients)} bags")
    
    # Re-apply our study habits for each split
    train_data = Subset(full_dataset, train_indices)
    val_data = Subset(full_dataset, val_indices)
    
    class TransformDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            # In bag_mode, img is a stacked tensor (Bag_Size, C, H, W)
            imgs, label, patient_id = self.subset[index]
            # Transforms are already applied in HPyloriDataset.__getitem__ via self.transform
            # but we can re-apply or wrap if needed. For now, HPyloriDataset handles it.
            return imgs, label, patient_id, index
        def __len__(self):
            return len(self.subset)

    # Use the transforms in the underlying dataset
    full_dataset.transform = train_transform
    train_transformed = TransformDataset(train_data, None) 
    
    # We'll switch transform for validation later or use a proxy
    val_transformed = TransformDataset(val_data, None)

    # --- Step 4.5: Improve Recall for Contaminated Samples ---
    # Distribution of bags
    train_labels = [full_dataset.bags[i][1] for i in train_indices]
    neg_count = train_labels.count(0)
    pos_count = train_labels.count(1)
    print(f"Training distribution (Bags): Negative={neg_count}, Contaminated={pos_count}")

    # Weighted Sampling for Bags
    class_weights = [1.0/max(1, neg_count), 1.0/max(1, pos_count)]
    sampler_weights = torch.FloatTensor([class_weights[t] for t in train_labels])
    sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights))

    # MIL Batch Size: Usually 1 bag per batch is safest for variable sizes, 
    # but we can try small batches if bag sizes are fixed/padded.
    # Given A40 VRAM and 500 max patches, batch_size=1 (bag-at-a-time) is mandatory.
    batch_size_mil = 1 
    accumulation_steps = 16 # Adjust for MIL bag-level steps
    train_loader = DataLoader(
        train_transformed, 
        batch_size=batch_size_mil, 
        sampler=sampler, 
        num_workers=4, # Reduced for bag loading overhead
        pin_memory=True
    )
    val_loader = DataLoader(
        val_transformed, 
        batch_size=batch_size_mil, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # --- Step 5: Build the customized AI brain ---
    model = get_model(model_name=model_name, num_classes=2, pretrained=True).to(device)

    # --- Step 6: Define the Learning Rules ---
    # strategy B: Focal Loss for Sparse Bacteremia Detection
    # Optimized to ignore common histological background and focus on sparse bacteria.
    # We invert weights to [1.5, 1.0] (Optimization 8.2) to recover specificity
    # from the 20-30% floor and provide cleaner features for aggregation.
    # Label Smoothing is disabled (0.0) to maximize bacterial signal contrast (8.4).
    loss_weights = torch.FloatTensor([1.5, 1.0]).to(device) 
    criterion = FocalLoss(gamma=2, weight=loss_weights, smoothing=0.0)

    # --- Optimization 5D: Preprocessing & Model Compilation (Kernel Fusion) ---
    # We define a fused preprocessing function for deterministic operations.
    # IHC Pivot: Disabling Macenko Normalization because dataset is IHC (Blue/Brown),
    # not H&E. Macenko was causing color collapse to black & white.
    def det_preprocess_batch(inputs, training=False):
        # inputs = normalizer.normalize_batch(inputs, jitter=training) # DISABLED for IHC
        inputs = gpu_normalize(inputs)
        return inputs
    
    # Enable torch.compile for Kernel Fusion (Optimization 5D)
    if hasattr(torch, "compile"):
        # Disabling compilation for MIL bags due to high VRAM overhead
        print(f"Skipping torch.compile for {model_name} to conserve VRAM...")
        # model = torch.compile(model, mode="reduce-overhead")
    
    # Optimizer Choice:
    # ConvNeXt is highly sensitive to the training recipe and benefits from AdamW.
    if "convnext" in model_name:
        print(f"Using AdamW Optimizer for {model_name} stability...")
        optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
    else:
        # ResNet default (Adam)
        optimizer = Adam(model.parameters(), lr=2e-5, weight_decay=5e-3)
    
    # --- Step 6.2: OneCycle Learning Rate Scheduler ---
    num_epochs = 1
    steps_per_epoch = len(train_loader) // accumulation_steps
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-4, 
        epochs=num_epochs, 
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1, # 10% Warmup phase
        anneal_strategy='cos'
    )
    
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
    # num_epochs = 15 // Set via scheduler in Step 6.2
    best_loss = float('inf')
    
    # Track the "History" to plot learning curves later
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    # Create a wrapper for autocast that identifies the actual device type
    def get_autocast_device():
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    # Suppress specific deprecation warnings from torch.utils.checkpoint
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
    
    # Global memory optimization for CUDA
    if torch.cuda.is_available():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # Reduce fraction further to 0.7 to ensure system can't over-allocate
        torch.cuda.set_per_process_memory_fraction(0.70, 0) 
        torch.cuda.empty_cache()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # --- Study Mode (Train) ---
        model.train() 
        full_dataset.transform = train_transform # Set training augmentations
        running_loss = 0.0
        running_corrects = 0
        optimizer.zero_grad()
        
        for i, (bags, labels, patient_ids, _) in enumerate(tqdm(train_loader, desc="Training (MIL Bag)")):
            # bags shape: (1, Bag_Size, C, H, W) -> (Bag_Size, C, H, W)
            bags = bags.squeeze(0).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # --- GPU-Based Preprocessing ---
            # Apply same deterministic normalize for IHC
            bags = det_preprocess_batch(bags, training=True)
            
            # Use dynamic device type for autocast to avoid warnings on CPU-only runs
            with torch.amp.autocast(device_type=get_autocast_device()):
                # Use the MIL forward pass
                outputs, _ = model.forward_bag(bags) # returns (1, 2), (1, N)
                loss = criterion(outputs, labels) 
                loss = loss / accumulation_steps
            
            _, preds = torch.max(outputs, 1)
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                # Clear cache after optimization step to prevent creep
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            running_loss += (loss.item() * accumulation_steps)
            running_corrects += torch.sum(preds == labels.data).item()
            
        epoch_loss = running_loss / len(train_indices)
        epoch_acc = float(running_corrects) / len(train_indices)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- Self-Test Mode (Validation) ---
        model.eval()
        full_dataset.train = False # No random sampling during eval
        full_dataset.transform = val_transform # No random augment during eval
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for i, (bags, labels, patient_ids, _) in enumerate(tqdm(val_loader, desc="Validation (MIL Bag)")):
                bags = bags.squeeze(0).to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                bags = det_preprocess_batch(bags, training=False)
                
                with torch.amp.autocast(device_type=get_autocast_device()):
                    outputs, _ = model.forward_bag(bags)
                    loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item()
                val_corrects += torch.sum(preds == labels.data).item()
                
        val_epoch_loss = val_loss / len(val_indices)
        val_epoch_acc = float(val_corrects) / len(val_indices)
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

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

    # --- Step 7.4: Memory Cleanup ---
    # Delete training loaders and clear cache to free up RAM for the Hold-Out test
    print(f"Fold {fold_idx} finished.")
    del train_loader
    del val_loader
    del train_transformed
    del val_transformed
    del full_dataset # Reclaim dataset memory
    del train_data
    del val_data
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

    # --- Step 8: Final Evaluation (MIL Bag Mode + 8-way TTA) ---
    print(f"\nEvaluating on Independent Hold-Out set from: {holdout_dir} with 8-way TTA and Multi-Pass Bag Coverage")
    
    # We set max_bag_size=10000 for evaluation to ensure we load the full bag
    # and then manually chunk it in the loop below to stay within VRAM limits.
    holdout_dataset = HPyloriDataset(holdout_dir, patient_csv, patch_csv, transform=val_transform, bag_mode=True, max_bag_size=10000, train=False)
    
    holdout_loader = DataLoader(
        holdout_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    # 8-way TTA transforms (Flips and Rotations)
    tta_transforms = [
        lambda x: x, # Original
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
        lambda x: torch.rot90(x, 1, [2, 3]),
        lambda x: torch.rot90(x, 2, [2, 3]),
        lambda x: torch.rot90(x, 3, [2, 3]),
        lambda x: v2.RandomHorizontalFlip(p=1.0)(torch.rot90(x, 1, [2, 3])),
        lambda x: v2.RandomVerticalFlip(p=1.0)(torch.rot90(x, 1, [2, 3]))
    ]

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    
    num_holdout = len(holdout_dataset)
    all_preds_pat = np.zeros(num_holdout, dtype=np.int8)
    all_labels_pat = np.zeros(num_holdout, dtype=np.int8)
    all_probs_pat = np.zeros(num_holdout, dtype=np.float32)
    
    patient_ids_list = []
    # Use a chunk size of 500 for the Attention aggregating loop to prevent OOM
    vram_bag_limit = 500

    with torch.no_grad():
        for i, (bags, labels, patient_ids) in enumerate(tqdm(holdout_loader, desc="Patient-Independent TTA Test")):
            # bags shape: (1, Bag_Size, C, H, W) -> (Bag_Size, C, H, W)
            bags = bags.squeeze(0).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Divide bag into chunks of 500 if larger
            bag_size = bags.size(0)
            bag_probs_list = []
            
            for start_idx in range(0, bag_size, vram_bag_limit):
                chunk_bags = bags[start_idx : start_idx + vram_bag_limit]
                
                # Aggregate TTA logits for this chunk
                chunk_logits_sum = None
                for tta_aug in tta_transforms:
                    aug_bags = tta_aug(chunk_bags)
                    aug_bags = det_preprocess_batch(aug_bags, training=False)
                    
                    with torch.amp.autocast(device_type=get_autocast_device()):
                        logits, _ = model.forward_bag(aug_bags)
                    
                    if chunk_logits_sum is None:
                        chunk_logits_sum = logits
                    else:
                        chunk_logits_sum += logits
                
                # Average TTA results for this chunk and store probability
                avg_chunk_logits = chunk_logits_sum / len(tta_transforms)
                chunk_probs = torch.softmax(avg_chunk_logits, dim=1)
                bag_probs_list.append(chunk_probs)
            
            # --- AGGREGATION: Average across chunks (Multi-Pass Voting) ---
            final_bag_probs = torch.stack(bag_probs_list).mean(0)
            preds = torch.argmax(final_bag_probs, dim=1)
            
            all_preds_pat[i] = preds.cpu().item()
            all_labels_pat[i] = labels.cpu().item()
            all_probs_pat[i] = final_bag_probs[:, 1].cpu().item()
            patient_ids_list.append(patient_ids[0])
            
            if i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    # --- Step 9: Detailed Reporting (Patient Level Focused) ---
    print("\nPatient-Level Classification Report (MIL):")
    print(classification_report(all_labels_pat, all_preds_pat, target_names=['Negative', 'Positive'], zero_division=0))
    
    # Save clinical consensus CSV (re-using old format for metrics/plots)
    consensus_data = []
    for i in range(num_holdout):
        consensus_data.append({
            "PatientID": patient_ids_list[i],
            "Actual": all_labels_pat[i],
            "Predicted": all_preds_pat[i],
            "Mean_Prob": all_probs_pat[i], # For MIL, the bag prob IS the diagnosis
            "Max_Prob": all_probs_pat[i],
            "Meta_Prob": all_probs_pat[i],
            "Patch_Count": 0, # Not used in MIL evaluation loop this way
            "Method": "Attention-MIL",
            "Correct": 1 if all_preds_pat[i] == all_labels_pat[i] else 0
        })
    consensus_df = pd.DataFrame(consensus_data)
    consensus_df.to_csv(os.path.join(results_dir, f"{prefix}_patient_consensus.csv"), index=False)

    # Step 11: Patient plots re-use
    fpr_meta, tpr_meta, _ = roc_curve(all_labels_pat, all_probs_pat)
    roc_auc_meta = auc(fpr_meta, tpr_meta)
    
    plt.figure()
    plt.plot(fpr_meta, tpr_meta, color='darkorange', lw=3, label=f'Attention-MIL (AUC = {roc_auc_meta:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Patient-Level ROC: Attention-MIL')
    plt.legend(loc="lower right")
    plt.savefig(patient_roc_path)
    plt.close()
    
    # Calculate Patient-Level Accuracy
    pat_acc = (all_preds_pat == all_labels_pat).mean() * 100
    print(f"\nFinal Patient-Level Accuracy (MIL): {pat_acc:.2f}%")

    # --- Step 12: Extra Visuals (Metrics & Interpretability) ---
    print("\nGenerating final metrics and interpretability maps...")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels_pat, all_preds_pat)
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive']).plot(cmap='Blues')
    plt.title('Patient-Level Confusion Matrix')
    plt.savefig(patient_cm_path)
    plt.close()

    # 2. Probability Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(all_probs_pat[all_labels_pat == 0], bins=20, alpha=0.5, label='Actual Negative', color='blue')
    plt.hist(all_probs_pat[all_labels_pat == 1], bins=20, alpha=0.5, label='Actual Positive', color='red')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Predicted Probability (Positive Class)')
    plt.ylabel('Patient Count')
    plt.title('Patient-Level Probability Distribution')
    plt.legend()
    plt.savefig(hist_path)
    plt.close()

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_labels_pat, all_probs_pat)
    avg_prec = average_precision_score(all_labels_pat, all_probs_pat)
    plt.figure()
    plt.plot(recall, precision, color='green', lw=2, label=f'PR (AP = {avg_prec:0.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Patient-Level PR Curve')
    plt.legend()
    plt.savefig(patient_pr_path)
    plt.close()

    # 4. Grad-CAM Samples from Hold-Out
    print(f"Generating Grad-CAM for most suspicious patient bags in {gradcam_dir}...")
    # Get top 5 most positive and top 5 most ghostly (FN)
    # top_indices: highest probability, fn_indices: highest prob among actual positive that were pred 0
    top_indices = np.argsort(all_probs_pat)[-5:] 
    
    # Target the last conv layer for Grad-CAM
    # For ConvNeXt-Tiny, it's typically the last block of the last stage
    if "convnext" in model_name:
        # ConvNeXt Features Stage 4 Block 3
        target_layer = model.backbone.features[7][2].block[5]
    else:
        target_layer = model.backbone.layer4[2]

    # Free up memory before visualization
    torch.cuda.empty_cache()
    
    # Context to enable gradients for Grad-CAM
    for bag_idx in top_indices:
        p_id = patient_ids_list[bag_idx]
        
        # Load full bag for this patient (keep on CPU initially)
        bag_imgs, label, _ = holdout_dataset[bag_idx]
        # Use chunks for forward pass - keep chunks on CPU, load only current chunk to GPU
        
        # Find Attention weights to choose the top patches (we don't need gradients for this part)
        all_attns = []
        with torch.no_grad():
            for start_idx in range(0, bag_imgs.size(0), vram_bag_limit):
                chunk = bag_imgs[start_idx:start_idx + vram_bag_limit].to(device)
                chunk = det_preprocess_batch(chunk, training=False)
                _, attn = model.forward_bag(chunk)
                all_attns.append(attn.cpu()) # Keep results on CPU to save VRAM
        
        attention = torch.cat(all_attns, dim=1).squeeze(0) # (Bag_Size,)
        
        # Take top 3 most "Attended" patches
        top_patch_vals, top_patch_indices = torch.topk(attention, k=min(3, bag_imgs.size(0)))
        
        for rank, p_idx in enumerate(top_patch_indices):
            # Only load the specific patch and its input tensor for Grad-CAM
            patch_img = bag_imgs[p_idx:p_idx+1].to(device)
            patch_input = det_preprocess_batch(patch_img, training=False)
            
            # Context to enable gradients for Grad-CAM on this specific patch
            with torch.enable_grad():
                heatmap, p_probs = generate_gradcam(model.backbone, patch_input, target_layer)
            
            # Plotting logic
            plt.figure(figsize=(10, 5))
            # Original Patch
            plt.subplot(1, 2, 1)
            orig_img = patch_img[0].cpu().permute(1, 2, 0).numpy()
            plt.imshow(orig_img)
            plt.title(f"Patch {p_idx} (Attn: {top_patch_vals[rank]:.4f})")
            plt.axis('off')
            
            # Heatmap
            plt.subplot(1, 2, 2)
            plt.imshow(orig_img)
            plt.imshow(heatmap[0, 0], cmap='jet', alpha=0.5)
            plt.title(f"Grad-CAM (Pos Prob: {p_probs[0, 1]:.4f})")
            plt.axis('off')
            
            out_path = os.path.join(gradcam_dir, f"{p_id}_rank{rank}_patch{p_idx}.png")
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            
            # Cleanup per patch
            del patch_img, patch_input, heatmap
            torch.cuda.empty_cache()

    # Repeat for False Negatives (Ghost Patients)
    fn_indices = [i for i, (prob, label) in enumerate(zip(all_probs_pat, all_labels_pat)) if label == 1 and prob < 0.5]
    fn_indices = sorted(fn_indices, key=lambda i: all_probs_pat[i], reverse=True)[:5]
    
    if fn_indices:
        print(f"Generating Grad-CAM for {len(fn_indices)} False Negative (Ghost) bags...")
        for bag_idx in fn_indices:
            p_id = patient_ids_list[bag_idx]
            bag_imgs, label, _ = holdout_dataset[bag_idx]
            
            all_attns = []
            with torch.no_grad():
                for start_idx in range(0, bag_imgs.size(0), vram_bag_limit):
                    chunk = bag_imgs[start_idx:start_idx + vram_bag_limit].to(device)
                    chunk = det_preprocess_batch(chunk, training=False)
                    _, attn = model.forward_bag(chunk)
                    all_attns.append(attn.cpu())
            
            attention = torch.cat(all_attns, dim=1).squeeze(0)
            top_patch_vals, top_patch_indices = torch.topk(attention, k=min(3, bag_imgs.size(0)))
            
            for rank, p_idx in enumerate(top_patch_indices):
                patch_img = bag_imgs[p_idx:p_idx+1].to(device)
                patch_input = det_preprocess_batch(patch_img, training=False)
                
                with torch.enable_grad():
                    heatmap, p_probs = generate_gradcam(model.backbone, patch_input, target_layer)
                
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                orig_img = patch_img[0].cpu().permute(1, 2, 0).numpy()
                plt.imshow(orig_img)
                plt.title(f"FN Patch {p_idx} (Attn: {top_patch_vals[rank]:.4f})")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(orig_img)
                plt.imshow(heatmap[0, 0], cmap='jet', alpha=0.5)
                plt.title(f"FN Grad-CAM (Pos Prob: {p_probs[0, 1]:.4f})")
                plt.axis('off')
                
                out_path = os.path.join(gradcam_dir, f"FN_{p_id}_rank{rank}_patch{p_idx}.png")
                plt.savefig(out_path, bbox_inches='tight')
                plt.close()
                
                del patch_img, patch_input, heatmap
                torch.cuda.empty_cache()
                
    # Save evaluation report
    report = classification_report(all_labels_pat, all_preds_pat, target_names=['Negative', 'Positive'], output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(results_csv_path)
    print(f"Evaluation report saved to {results_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H. Pylori K-Fold Training")
    parser.add_argument("--fold", type=int, default=0, help="Index of the fold to use for validation (0 to num_folds-1)")
    parser.add_argument("--num_folds", type=int, default=5, help="Total number of folds")
    parser.add_argument("--model_name", type=str, default="convnext_tiny", choices=["resnet50", "convnext_tiny"], 
                        help="Backbone architecture to use (resnet50/convnext_tiny)")
    args = parser.parse_args()
    
    train_model(fold_idx=args.fold, num_folds=args.num_folds, model_name=args.model_name)
