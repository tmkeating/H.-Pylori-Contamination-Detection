import os                       # Standard library for file path management
import torch                    # Core library for deep learning
import torch.nn as nn           # Tools for building neural network layers
import torch.optim as optim     # Mathematical tools to "teach" the model
from torch.utils.data import DataLoader, random_split # Tools to manage and split data
from torchvision import transforms # Tools to prep images for the AI
from dataset import HPyloriDataset # Our custom code that finds images/labels
from model import get_model        # Our custom code that builds the AI brain
from tqdm import tqdm              # A library that shows a "progress bar"

def train_model():
    # --- Step 1: Choose our study device ---
    # Use a Graphics Card (CUDA) if available; otherwise, use the Main Processor (CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 2: Set the paths to our data ---
    patient_csv = "../HelicoDataSet/PatientDiagnosis.csv" # General results
    patch_csv = "../HelicoDataSet/HP_WSI-CoordAnnotatedAllPatches.csv" # Specific spot results
    train_dir = "../HelicoDataSet/CrossValidation/Annotated" # Folder used for training
    holdout_dir = "../HelicoDataSet/HoldOut" # Folder used for the final "final exam"

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
    # Create the full dataset object
    full_dataset = HPyloriDataset(train_dir, patient_csv, patch_csv, transform=train_transform)
    
    # Split the main data: 80% for studying (training), 20% for self-testing (validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    # Ensure the self-test data doesn't use the "flipping" study habits
    val_data.dataset.transform = val_transform

    # --- Step 4.5: Improve Recall for Contaminated Samples ---
    # We calculate the distribution of our training data
    train_indices = train_data.indices
    train_labels = [full_dataset.samples[i][1] for i in train_indices]
    neg_count = train_labels.count(0)
    pos_count = train_labels.count(1)
    print(f"Training distribution: Negative={neg_count}, Contaminated={pos_count}")

    # strategy A: Weighted Sampling (Oversampling)
    # This ensures that every batch of 32 images is balanced (approx 16 Neg, 16 Pos)
    class_weights = [1.0/neg_count, 1.0/pos_count]
    sample_weights = [class_weights[t] for t in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    # DataLoaders are like "librarians" that hand the AI images in batches of 32
    # We use the 'sampler' to show contaminated images more frequently to the AI
    train_loader = DataLoader(train_data, batch_size=32, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)

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
    
    # Optional Step: Use special Intel optimization if possible for faster speed
    try:
        import intel_extension_for_pytorch as ipex
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
        print("Intel Extension for PyTorch optimization enabled.")
    except Exception as e:
        print(f"IPEX stabilization/matching skipped: {e}. Running on standard PyTorch.")

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
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
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
                val_corrects += torch.sum(preds == labels.data)
                
        val_epoch_loss = val_loss / val_size
        val_epoch_acc = val_corrects.double() / val_size
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
    print("\nEvaluating on HoldOut set...")
    holdout_dataset = HPyloriDataset(holdout_dir, patient_csv, patch_csv, transform=val_transform)
    holdout_loader = DataLoader(holdout_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load our best saved brain
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    
    h_corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(holdout_loader, desc="HoldOut"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            h_corrects += torch.sum(preds == labels.data)
            
    print(f"HoldOut Accuracy: {h_corrects.double() / len(holdout_dataset):.4f}")

if __name__ == "__main__":
    train_model() # Run the whole process start to finish
