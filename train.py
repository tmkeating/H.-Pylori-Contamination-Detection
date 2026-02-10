import os                       # Standard library for file path management
import torch                    # Core library for deep learning
import torch.nn as nn           # Tools for building neural network layers
import torch.optim as optim     # Mathematical tools to "teach" the model
from torch.optim.adam import Adam # The specific algorithm to adjust the brain
import numpy as np               # Numeric library
import pandas as pd              # Data manipulation library
import matplotlib.pyplot as plt  # Drawing/plotting library
from torch.utils.data import DataLoader, random_split, Dataset, Subset, WeightedRandomSampler # Tools to manage and split data
from torchvision import transforms # Tools to prep images for the AI
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, 
    classification_report, precision_recall_curve, 
    average_precision_score, PrecisionRecallDisplay
) 
from dataset import HPyloriDataset # Our custom code that finds images/labels
from model import get_model        # Our custom code that builds the AI brain
from tqdm import tqdm              # A library that shows a "progress bar"
import re                          # Regexp to handle file numbering
import torch.nn.functional as F

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

def get_next_run_number(results_dir="results"):
    """Finds the next available numeric prefix for output files."""
    if not os.path.exists(results_dir):
        return 0
    
    files = os.listdir(results_dir)
    prefixes = []
    for f in files:
        match = re.match(r"^(\d+)_", f)
        if match:
            prefixes.append(int(match.group(1)))
    
    return max(prefixes) + 1 if prefixes else 0

# This helper class allows us to have different transforms for train and validation split
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        # The base dataset usually provides raw images if no transform is passed to it,
        # but here the original HPyloriDataset might already have a transform.
        # We access the image and label from the subset
        # We need to reach the original image data without the full_dataset's transform.
        # Actually, if we just want to override, we should create the full_dataset with transform=None.
        img, label = self.subset[index] # This currently uses train_transform because full_dataset has it
        # If we really want to swap transforms, we should set full_dataset.transform = None 
        # and apply them here.
        return img, label

    def __len__(self):
        return len(self.subset)

def train_model():
    # --- Step 0: Prepare output directories ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get the numeric run ID and the SLURM job ID (if it exists)
    run_id = f"{get_next_run_number(results_dir):02d}"
    slurm_id = os.environ.get("SLURM_JOB_ID", "local")
    prefix = f"{run_id}_{slurm_id}"
    print(f"--- Starting Run ID: {run_id} (SLURM Job: {slurm_id}) ---")

    # Define versioned file paths
    best_model_path = os.path.join(results_dir, f"{prefix}_model_brain.pth")
    results_csv_path = os.path.join(results_dir, f"{prefix}_evaluation_report.csv")
    cm_path = os.path.join(results_dir, f"{prefix}_confusion_matrix.png")
    roc_path = os.path.join(results_dir, f"{prefix}_roc_curve.png")
    pr_path = os.path.join(results_dir, f"{prefix}_pr_curve.png")
    history_path = os.path.join(results_dir, f"{prefix}_learning_curves.png")
    hist_path = os.path.join(results_dir, f"{prefix}_probability_histogram.png")
    gradcam_dir = os.path.join(results_dir, f"{prefix}_gradcam_samples")
    os.makedirs(gradcam_dir, exist_ok=True)

    # --- Step 1: Choose our study device ---
    # Use a Graphics Card (CUDA) if available; otherwise, use the Main Processor (CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 2: Set the paths to our data ---
    # The dataset is located on the shared cluster path
    base_data_path = "/import/fhome/vlia/HelicoDataSet"
    if not os.path.exists(base_data_path):
        # Fallback for local development or different environments
        local_path = "/home/twyla/Documents/Classes/aprenentatgeProfund/Code/HelicoDataSet"
        if os.path.exists(local_path):
            base_data_path = local_path
        else:
            # Last resort: look in the parent directory
            base_data_path = os.path.abspath(os.path.join(os.getcwd(), "..", "HelicoDataSet"))
        
        print(f"Primary path not found. Using: {base_data_path}")

    patient_csv = os.path.join(base_data_path, "PatientDiagnosis.csv")
    patch_csv = os.path.join(base_data_path, "HP_WSI-CoordAnnotatedAllPatches.xlsx") # Use Excel directly
    train_dir = os.path.join(base_data_path, "CrossValidation/Annotated")
    # Note: We will split the train_dir itself to ensure we have positive samples in evaluation
    holdout_dir = os.path.join(base_data_path, "HoldOut")

    # --- Step 3: Define "Study Habits" (Transforms) ---
    # Training habits: We resize and add variety to make the AI more robust
    train_transform = transforms.Compose([
        transforms.Resize((448, 448)), # Increased resolution to see tiny bacteria better
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15), # Small rotations to handle slide orientation
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Handle stain variation
        transforms.ToTensor(), 
        # Standardize colors so they are easier for the AI to understand
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation habits: No random variety here, just high resolution
    val_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- Step 4: Load and split the data ---
    # Create the full dataset object without a transform initially
    full_dataset = HPyloriDataset(train_dir, patient_csv, patch_csv, transform=None)
    
    # Split the main data: 80% for training, 20% for validation
    indices = list(range(len(full_dataset)))
    train_size = int(0.8 * len(full_dataset))
    
    # Shuffle indices manually for the split
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    
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
            img, label = self.subset[index]
            if self.transform:
                img = self.transform(img)
            return img, label
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
    sample_weights = [class_weights[t] for t in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # DataLoaders are like "librarians" that hand the AI images in batches of 32
    # We use the 'sampler' to show contaminated images more frequently to the AI
    train_loader = DataLoader(train_transformed, batch_size=32, sampler=sampler, num_workers=8)
    val_loader = DataLoader(val_transformed, batch_size=32, shuffle=False, num_workers=8)

    # --- Step 5: Build the customized AI brain ---
    model = get_model(num_classes=2, pretrained=True).to(device)

    # --- Step 6: Define the Learning Rules ---
    # strategy B: Weighted Loss Function
    # We assign a higher penalty (e.g., 5x) for missing a contaminated sample.
    # This is critical now that we have a 50:1 imbalance.
    loss_weights = torch.FloatTensor([1.0, 3.0]).to(device) 
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    # Optimizer: The "tutor" (the math that updates the model to make it less wrong)
    optimizer = Adam(model.parameters(), lr=1e-4)
    
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
    # We will go through the entire set of images 10 times (10 "Epochs")
    num_epochs = 10
    best_loss = float('inf')
    
    # Track the "History" to plot learning curves later
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # --- Study Mode (Train) ---
        model.train() # Tell the brain it is in learning mode
        running_loss = 0.0
        running_corrects = 0
        
        # Hand batches of images to the AI one by one
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device) # Move images to CPU/GPU
            labels = labels.to(device) # Move correct answers to CPU/GPU
            
            optimizer.zero_grad() # Clear previous math notes
            outputs = model(inputs) # AI makes its guess
            loss = criterion(outputs, labels) # See how wrong the guess was
            _, preds = torch.max(outputs, 1) # Pick the best class (0 or 1)
            
            loss.backward() # Mathematical "backtracking" to see what to fix
            optimizer.step() # Tutor updates the brain's connections
            
            # Keep track of scores
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            
        epoch_loss = running_loss / train_size
        epoch_acc = float(running_corrects) / train_size
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- Self-Test Mode (Validation) ---
        model.eval() # Tell brain it's testing time (no more updating connections)
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad(): # Don't take any math notes, just grade
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                
        val_epoch_loss = val_loss / (len(full_dataset) - train_size)
        val_epoch_acc = float(val_corrects) / (len(full_dataset) - train_size)
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

    # --- Step 8: The Final Exam (HoldOut Test) ---
    # This is a set of data the AI has NEVER seen before during training
    print("\nEvaluating on HoldOut set (The Final Exam)...")
    # Using validation set as the "Final Exam" since it contains verified positive labels
    holdout_loader = DataLoader(val_transformed, batch_size=32, shuffle=False, num_workers=8)
    
    # Load our best saved brain
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds = []   # List to store the AI's final guesses (0 or 1)
    all_labels = []  # List to store the actual correct answers
    all_probs = []   # List to store how "sure" the AI was (e.g., 0.95 sure it's positive)
    gradcam_samples = [] # Store a few images for visualization

    with torch.no_grad():
        for inputs, labels in tqdm(holdout_loader, desc="Testing HoldOut"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs) # Get raw brain output
            probs = torch.softmax(outputs, dim=1) # Convert output to 100% probabilities
            _, preds = torch.max(outputs, 1) # Pick the highest probability class
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1]) # Probability of being "Contaminated"
            
            # Save a few contaminated samples for Grad-CAM
            if len(gradcam_samples) < 5:
                pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
                if len(pos_indices) > 0:
                    idx = pos_indices[0].item()
                    gradcam_samples.append(inputs[idx:idx+1].clone())

    # --- Step 9: Detailed Reporting ---
    
    # 1. Classification Report (Precision, Recall, F1)
    print("\nClassification Report:")
    report_dict = classification_report(all_labels, all_preds, target_names=['Negative', 'Contaminated'], output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=['Negative', 'Contaminated']))

    # 2. Save machine-readable CSV for AI evaluation
    results_df = pd.DataFrame(report_dict).transpose()
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    results_df.loc['overall_auc', 'precision'] = roc_auc 
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
        print("Generating Grad-CAM interpretability maps...")
        # For ResNet18, the last conv layer is usually layer4
        target_layer = model.layer4[-1]
        for i, img_tensor in enumerate(gradcam_samples):
            with torch.enable_grad(): # Grad-CAM needs gradients
                cam, prob = generate_gradcam(model, img_tensor, target_layer)
            
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
            plt.title(f"Original Path (Prob: {prob[0,1]:.2f})")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(img)
            plt.imshow(cam, cmap='jet', alpha=0.5)
            plt.title("Grad-CAM Heatmap")
            plt.axis('off')
            
            plt.savefig(os.path.join(gradcam_dir, f"sample_{i}.png"))
            plt.close()
        print(f"Saved Grad-CAM samples to {gradcam_dir}")

if __name__ == "__main__":
    train_model() # Run the whole process start to finish
