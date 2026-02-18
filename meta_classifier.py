import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

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
            "Patch_Count",
            "Spatial_Clustering" # New: Avg neighbors for high-conf patches
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
        y_true, y_pred = [], []
        
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Use smaller params for quicker cross-val estimation
            clf = RandomForestClassifier(n_estimators=50, max_depth=4, class_weight="balanced")
            clf.fit(X_train, y_train)
            y_true.extend(y_test)
            y_pred.extend(clf.predict(X_test))
            
        print("\n--- Meta-Classifier (Leave-One-Out) Results ---")
        print(classification_report(y_true, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

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
        
        return preds, reliability

if __name__ == "__main__":
    # Orchestration: Automatically find all historical data and rebuild the meta-classifier.
    import glob
    csv_files = glob.glob("results/*_patient_consensus.csv") + glob.glob("finalResults/*_patient_consensus.csv")
    
    meta = HPyMetaClassifier()
    try:
        # 1. Aggregate historical statistics
        data = meta.prepare_data(csv_files)
        # 2. Perform clinical cross-validation
        meta.evaluate(data)
        # 3. Train final model for deployment in train.py
        meta.train(data)
    except Exception as e:
        print(f"Could not build meta-classifier: {e}")
        print("Note: This likely means results files are in the old Iteration 1 format.")
