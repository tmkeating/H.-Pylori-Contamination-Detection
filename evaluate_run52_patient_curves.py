
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# --- Config ---
RUN_ID = "52_102093"
CSV_PATH = f"finalResults/{RUN_ID}_patient_consensus.csv"

def generate_patient_curves():
    print(f"--- Generating Patient-Level Curves for Run {RUN_ID} ---")
    
    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    
    # Convert string labels to binary
    # Actual: 'Positive' -> 1, 'Negative' -> 0
    y_true = (df['Actual'] == 'Positive').astype(int)
    
    # We will test two scoring methods for the curves:
    # A) Mean Probability (Standard average across tissue)
    # B) Max Probability (High-sensitivity outlier detection)
    # C) Suspicious Count (Density of patches > 0.90)
    
    methods = {
        'Average Probability': df['Mean_Prob'],
        'Max Probability': df['Max_Prob'],
        'Suspicious Count (>0.90)': df['Suspicious_Count']
    }
    
    plt.figure(figsize=(12, 5))
    
    # Plot ROC Curves
    plt.subplot(1, 2, 1)
    for name, scores in methods.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Patient-Level ROC Curves')
    plt.legend(loc="lower right")
    
    # Plot PR Curves
    plt.subplot(1, 2, 2)
    for name, scores in methods.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(recall, precision, lw=2, label=f'{name} (AP = {ap:.2f})')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Patient-Level Precision-Recall Curves')
    plt.legend(loc="lower left")
    
    output_path = f"{RUN_ID}_patient_evaluation_curves.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved patient-level curves to {output_path}")
    
    # Identify which method is best
    best_auc = 0
    best_method = ""
    for name, scores in methods.items():
        _, _, _ = roc_curve(y_true, scores)
        current_auc = auc(*roc_curve(y_true, scores)[:2])
        if current_auc > best_auc:
            best_auc = current_auc
            best_method = name
            
    print(f"\nRecommended Consensus Score: {best_method} with AUC: {best_auc:.2f}")

if __name__ == "__main__":
    generate_patient_curves()
