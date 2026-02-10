import os                       # Standard library for file path management
import torch                    # Core library for deep learning
import torch.nn as nn           # Tools for building neural network layers
import torch.optim as optim     # Mathematical tools to "teach" the model
import numpy as np               # Numeric library
import pandas as pd              # Data manipulation library
import matplotlib.pyplot as plt  # Drawing/plotting library
from torch.utils.data import DataLoader, random_split # Tools to manage and split data
from torchvision import transforms # Tools to prep images for the AI
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report 
from dataset import HPyloriDataset # Our custom code that finds images/labels
from model import get_model        # Our custom code that builds the AI brain
from tqdm import tqdm              # A library that shows a "progress bar"

# This helper class allows us to have different transforms for train and validation split
class TransformedSubset(torch.utils.data.Dataset):
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
    # --- Step 1: Choose our study device ---
    # Use a Graphics Card (CUDA) if available; otherwise, use the Main Processor (CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 2: Set the paths to our data ---
    patient_csv = "/home/twyla/Documents/Classes/aprenentatgeProfund/Code/HelicoDataSet/PatientDiagnosis.csv"
    patch_csv = "/home/twyla/Documents/Classes/aprenentatgeProfund/Code/HelicoDataSet/HP_WSI-CoordAnnotatedAllPatches.csv"
    train_dir = "/home/twyla/Documents/Classes/aprenentatgeProfund/Code/HelicoDataSet/CrossValidation/Annotated"
    holdout_dir = "/home/twyla/Documents/Classes/aprenentatgeProfund/Code/HelicoDataSet/HoldOut"

    # --- Step 3: Define "Study Habits" (Transforms) ---
    # Training habits: We resize and sometimes flip the image to make the AI more robust
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Make every image the same size
        transforms.RandomHorizontalFlip(), # Flip it sideways (helps model generalize)
        transforms.RandomVerticalFlip(), # Flip it upside down
        transforms.ToTensor(), # Convert the image into numbers (a "tensor")
        # Standardize colors so they are easier for the AI to understand
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation habits: No flipping here, we want to see the images as they really are
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
    train_data = torch.utils.data.Subset(full_dataset, train_indices)
    val_data = torch.utils.data.Subset(full_dataset, val_indices)
    
    # We assign the transforms to the original dataset temporarily during retrieval
    # or better, we use a custom class to apply them
    class TransformDataset(torch.utils.data.Dataset):
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
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    # DataLoaders are like "librarians" that hand the AI images in batches of 32
    # We use the 'sampler' to show contaminated images more frequently to the AI
    train_loader = DataLoader(train_transformed, batch_size=32, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_transformed, batch_size=32, shuffle=False, num_workers=4)

    # --- Step 5: Build the customized AI brain ---
    model = get_model(num_classes=2, pretrained=True).to(device)

    # --- Step 6: Define the Learning Rules ---
    # strategy B: Weighted Loss Function
    # We assign a higher penalty (e.g., 5x) for missing a contaminated sample.
    # This is standard practice in viral/bacterial detection where missing a case is critical.
    loss_weights = torch.FloatTensor([1.0, 5.0]).to(device) 
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    # Optimizer: The "tutor" (the math that updates the model to make it less wrong)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
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
    best_acc = 0.0

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

        # --- Report Card: Save the best version ---
        # If this epoch was the best yet, save the brain's state to a file
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved.")

    print(f"Training complete. Best Val Acc: {best_acc:.4f}")

    # --- Step 8: The Final Exam (HoldOut Test) ---
    # This is a set of data the AI has NEVER seen before during training
    print("\nEvaluating on HoldOut set (The Final Exam)...")
    holdout_dataset = HPyloriDataset(holdout_dir, patient_csv, patch_csv, transform=val_transform)
    holdout_loader = DataLoader(holdout_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load our best saved brain
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    
    all_preds = []   # List to store the AI's final guesses (0 or 1)
    all_labels = []  # List to store the actual correct answers
    all_probs = []   # List to store how "sure" the AI was (e.g., 0.95 sure it's positive)

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
    results_df.to_csv("evaluation_report_ai.csv")
    print("Saved machine-readable report to evaluation_report_ai.csv")

    # 3. Save Confusion Matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Contaminated'])
    disp.plot(cmap='Blues') # Use the string name to avoid linter confusion
    plt.title("HoldOut Set: Confusion Matrix")
    plt.savefig("confusion_matrix_final.png")
    print("Saved confusion_matrix_final.png")

    # 4. Save ROC Curve plot
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve_final.png")
    print("Saved roc_curve_final.png")

if __name__ == "__main__":
    train_model() # Run the whole process start to finish
