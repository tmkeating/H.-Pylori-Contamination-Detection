import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score, ConfusionMatrixDisplay
)
import joblib
import os
import matplotlib.pyplot as plt

class HPyMetaClassifier:
    """
    Distributional Meta-Classifier for H. pylori detection.
    
    This secondary diagnostic layer does not analyze images directly. Instead, it 
    evaluates the probabilistic 'signature' produced by the ResNet50 for each patient
    to distinguish between true bacterial signals and histological artifacts.
    """
    def __init__(self, model_path="meta_rf_classifier.joblib"):
        self.model_path = model_path
        
        # The 'Patient Signature': 18 variables characterizing the patch distribution.
        self.features = [
            # Central Tendency & Spread:
            "Mean_Prob", "Max_Prob", "Min_Prob", "Std_Prob", "Median_Prob",
            
            # Distributional Shape (Percentiles):
            "P10_Prob", "P25_Prob", "P75_Prob", "P90_Prob", 
            
            # Higher-Order Moments:
            "Skew", "Kurtosis",
            
            # Density Thresholds:
            "Count_P50", "Count_P60", "Count_P70", "Count_P80", "Count_P90",
            
            # Exposure & Spatial Context:
            "Patch_Count"
        ]
        
        # RandomForest Ensemble: 
        # Uses 100 decision trees to reach a diagnostic consensus.
        # max_depth=5 prevents overfitting on smaller clinical datasets.
        # class_weight='balanced' handles prevalence imbalance (Negative vs Positive slides).
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight="balanced",
            random_state=42
        )

    def prepare_data(self, csv_paths):
        """
        Scans results/ and finalResults/ for patient consensus CSVs.
        Validates that files contain the 17-dimensional signature before merging.
        """
        dfs = []
        for path in csv_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                # Ensure all features exist for clinical consistency
                if set(self.features).issubset(df.columns):
                    dfs.append(df)
        
        if not dfs:
            raise ValueError("No valid consensus data found with required features. Run a training session first.")
        
        return pd.concat(dfs, ignore_index=True)

    def tune_hyperparameters(self, data):
        """
        Performs a Grid Search over Random Forest parameters using LOPO-CV.
        
        This finds the configuration that best separates bacteria from artifacts
        by considering thousands of potential decision pathways.
        """
        X = data[self.features]
        y = data["Actual"]
        groups = data["PatientID"]
        
        logo = LeaveOneGroupOut()
        
        # Define the sweep range (Iteration 9.1: Clinical Grid Sweep)
        param_grid = {
            'n_estimators': [100, 200, 400],
            'max_depth': [3, 5, 8, None],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        print("\n--- Starting Meta-Classifier Hyperparameter Sweep (LOPO-CV) ---")
        base_rf = RandomForestClassifier(class_weight="balanced", random_state=42)
        
        # We use accuracy to align with the project's '92% Barrier' goal
        grid_search = GridSearchCV(
            estimator=base_rf,
            param_grid=param_grid,
            cv=logo,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y, groups=groups)
        
        print(f"Best Configuration: {grid_search.best_params_}")
        print(f"Grid Cross-Val Accuracy: {grid_search.best_score_:.4f}")
        
        # Update the class model with the best found parameters
        self.rf = grid_search.best_estimator_
        return grid_search.best_params_

    def train(self, data):
        """
        Performs the final supervised learning pass on all available patient data.
        Serializes the model to disk for deployment in the inference pipeline.
        """
        X = data[self.features]
        y = data["Actual"]
        
        print(f"Training Meta-Classifier on {len(data)} patients...")
        self.rf.fit(X, y)
        joblib.dump(self.rf, self.model_path)
        print(f"Meta-Classifier saved to {self.model_path}")

        # Save Feature Importance Report
        importances = self.rf.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        feat_path = "results/meta_feature_importance.csv"
        # Ensure results/ exists
        os.makedirs("results", exist_ok=True)
        feat_df.to_csv(feat_path, index=False)
        print(f"Feature Importance saved to {feat_path}")

        # Plot Feature Importance
        plt.figure(figsize=(10, 6))
        plt.barh(feat_df['Feature'][::-1], feat_df['Importance'][::-1], color='steelblue')
        plt.xlabel('Importance Weight')
        plt.title('Meta-Classifier: Clinical Feature Importance')
        plt.tight_layout()
        plt.savefig("results/meta_feature_importance.png")
        plt.close()

    def evaluate(self, data):
        """
        Performs Leave-One-Patient-Out Cross-Validation (LOPO-CV).
        
        This is the clinical gold standard: we train a model on N-1 patients and
        diagnose the 'missing' one. This ensures the model does not memorize
        stain-batch specific artifacts unique to a single patient folder.
        """
        X = data[self.features]
        y = data["Actual"]
        groups = data["PatientID"]
        
        logo = LeaveOneGroupOut()
        y_true, y_pred, y_probs = [], [], []
        
        print("\nRunning Leave-One-Patient-Out Cross-Validation...")
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Use the parameters found during the Hyperparameter Sweep
            params = self.rf.get_params()
            clf = RandomForestClassifier(**params)
            clf.fit(X_train, y_train)
            
            y_true.extend(y_test)
            y_pred.extend(clf.predict(X_test))
            y_probs.extend(clf.predict_proba(X_test)[:, 1])
            
        print("\n--- Meta-Classifier (Leave-One-Out) Results ---")
        report = classification_report(y_true, y_pred, output_dict=True)
        print(classification_report(y_true, y_pred))

        # Save Performance Metrics
        metrics_df = pd.DataFrame(report).transpose()
        metrics_df.to_csv("results/meta_performance_metrics.csv")
        print(f"Performance metrics saved to results/meta_performance_metrics.csv")

        # 1. Confusion Matrix Plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Meta-Classifier: Patient-Level Confusion Matrix')
        plt.savefig("results/meta_confusion_matrix.png")
        plt.close()

        # 2. ROC Curve Plot
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        # Baseline: Max Probability (Backbone only)
        fpr_max, tpr_max, _ = roc_curve(data["Actual"], data["Max_Prob"])
        auc_max = auc(fpr_max, tpr_max)
        
        # Baseline: Suspicious Count (Count_P90)
        fpr_susp, tpr_susp, _ = roc_curve(data["Actual"], data["Count_P90"])
        auc_susp = auc(fpr_susp, tpr_susp)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'Meta-Classifier (AUC = {roc_auc:0.2f})')
        plt.plot(fpr_max, tpr_max, color='green', lw=2, linestyle='--', label=f'Max Probability (AUC = {auc_max:0.2f})')
        plt.plot(fpr_susp, tpr_susp, color='purple', lw=2, linestyle=':', label=f'Suspicious Count (AUC = {auc_susp:0.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Meta-Classifier: ROI Analysis')
        plt.legend(loc="lower right")
        plt.savefig("results/meta_roc.png")
        plt.close()

        # 3. Precision-Recall Curve Plot
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        ap_score = average_precision_score(y_true, y_probs)
        
        # Max Prob PR
        prec_max, rec_max, _ = precision_recall_curve(data["Actual"], data["Max_Prob"])
        ap_max = average_precision_score(data["Actual"], data["Max_Prob"])
        
        # Suspicious Count PR
        prec_susp, rec_susp, _ = precision_recall_curve(data["Actual"], data["Count_P90"])
        ap_susp = average_precision_score(data["Actual"], data["Count_P90"])

        plt.figure()
        plt.plot(recall, precision, color='blue', lw=3, label=f'Meta-Classifier (AP = {ap_score:0.2f})')
        plt.plot(rec_max, prec_max, color='green', lw=2, linestyle='--', label=f'Max Probability (AP = {ap_max:0.2f})')
        plt.plot(rec_susp, prec_susp, color='purple', lw=2, linestyle=':', label=f'Suspicious Count (AP = {ap_susp:0.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Meta-Classifier: Precision-Recall (Artifact vs Signal)')
        plt.legend(loc="upper right")
        plt.savefig("results/meta_pr.png")
        plt.close()
        
        print("Meta-level plots (ROC, PR, CM) saved to results/")

    def predict(self, patient_stats_df):
        """
        Inference Deployment:
        Predicts labels and calculates a Reliability Score (Confidence).
        
        Reliability calculation:
        Tracks how far the ensemble probability is from the 0.5 decision boundary.
        1.0 = 100% ensemble consensus.
        0.0 = 50/50 split (Ambiguous case - flags for manual review).
        """
        if not os.path.exists(self.model_path):
            print("Warning: Meta-model not found. Falling back to heuristic gates.")
            return None
        
        clf = joblib.load(self.model_path)
        probs = clf.predict_proba(patient_stats_df[self.features])
        preds = clf.predict(patient_stats_df[self.features])
        
        # Reliability = np.abs(prob_positive - 0.5) * 2
        reliability = np.abs(probs[:, 1] - 0.5) * 2 
        
        return preds, reliability, probs[:, 1]

if __name__ == "__main__":
    # Orchestration: Automatically find all historical data and rebuild the meta-classifier.
    import glob
    csv_files = glob.glob("results/*_patient_consensus.csv") + glob.glob("finalResults/*_patient_consensus.csv")
    
    meta = HPyMetaClassifier()
    try:
        # 1. Aggregate historical statistics
        data = meta.prepare_data(csv_files)
        # 2. Perform Hyperparameter Sweep (Grid Search)
        meta.tune_hyperparameters(data)
        # 3. Perform detailed clinical cross-validation
        meta.evaluate(data)
        # 4. Train final model for deployment in train.py
        meta.train(data)
    except Exception as e:
        print(f"Could not build meta-classifier: {e}")
        print("Note: This likely means results files are in the old Iteration 1 format.")
