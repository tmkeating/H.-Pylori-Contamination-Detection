import torch # Deep learning framework
import torch.nn as nn # Neural network components
from torch.utils.data import DataLoader # Handles batch loading of data
from torchvision import transforms # Image preprocessing tools
import pandas as pd # Data manipulation library
import matplotlib.pyplot as plt # Library for creating charts and graphs
import numpy as np # Library for handling large numeric arrays
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report # Statistics tools
from dataset import HPyloriDataset # Our custom data loader
from model import get_model # Our custom AI brain builder
from tqdm import tqdm # Progress bar library

# This function runs the AI on new data and generates performance reports
def evaluate_model():
    # --- Step 1: Set up the environment ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths to our data and the saved "Best Brain"
    patient_csv = "/home/twyla/Documents/Classes/aprenentatgeProfund/Code/HelicoDataSet/PatientDiagnosis.csv"
    patch_csv = "/home/twyla/Documents/Classes/aprenentatgeProfund/Code/HelicoDataSet/HP_WSI-CoordAnnotatedAllPatches.csv"
    holdout_dir = "/home/twyla/Documents/Classes/aprenentatgeProfund/Code/HelicoDataSet/HoldOut"
    model_path = "/home/twyla/Documents/Classes/aprenentatgeProfund/Code/hPylori/best_model.pth"

    # Define how to prep the images (must match what we did during training)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- Step 2: Load the "Final Exam" data ---
    holdout_dataset = HPyloriDataset(holdout_dir, patient_csv, patch_csv, transform=val_transform)
    holdout_loader = DataLoader(holdout_dataset, batch_size=32, shuffle=False, num_workers=4)

    # --- Step 3: Load the saved AI brain ---
    model = get_model(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Put the brain in testing/inference mode

    all_preds = []   # List to store the AI's final guesses (0 or 1)
    all_labels = []  # List to store the actual correct answers
    all_probs = []   # List to store how "sure" the AI was (e.g., 0.95 sure it's positive)

    # --- Step 4: Run the test ---
    print("Running inference on HoldOut set...")
    with torch.no_grad(): # Don't update the brain, just perform the test
        for inputs, labels in tqdm(holdout_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs) # Get raw brain output
            probs = torch.softmax(outputs, dim=1) # Convert output to 0% - 100% probabilities
            _, preds = torch.max(outputs, 1) # Pick the highest probability class
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1]) # Store only the probability of being "Contaminated"

    # --- Step 5: Generate the Reports ---

    # Report 1: Classification Report (Precision, Recall, F1)
    print("\nClassification Report:")
    # This shows how accurate the model was for EACH category (Negative vs positive)
    report_dict = classification_report(all_labels, all_preds, target_names=['Negative', 'Contaminated'], output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=['Negative', 'Contaminated']))

    # --- Step 6: Save machine-readable results for AI evaluation ---
    # Convert the stats into a data table (DataFrame)
    results_df = pd.DataFrame(report_dict).transpose()
    
    # Calculate more metrics for the CSV
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    # Add the AUC score to the table for Gemini/AI analysis
    results_df.loc['overall_auc', 'precision'] = roc_auc 
    
    csv_filename = "evaluation_report_ai.csv"
    results_df.to_csv(csv_filename)
    print(f"Saved machine-readable report to {csv_filename}")

    # Report 2: Confusion Matrix
    # This is a grid showing True Positives, False Positives, etc.
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Contaminated'])
    disp.plot(cmap=plt.cm.Blues) # Plot it with a blue color scale
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png") # Save it as a picture file
    print("Saved confusion_matrix.png")

    # Report 3: ROC Curve and AUC
    # This measures how good the model is at separating the two classes overall
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Baseline (random guessing)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png") # Save it as a picture file
    print("Saved roc_curve.png")
    
    plt.show() # Show the final plots on the screen

if __name__ == "__main__":
    evaluate_model() # Execute the evaluation
