import glob
import pandas as pd
import joblib
from meta_classifier import HPyMetaClassifier

def replot():
    print("Collecting patient consensus data...")
    csv_files = glob.glob("results/*_patient_consensus.csv") + glob.glob("finalResults/*_patient_consensus.csv")
    
    meta = HPyMetaClassifier()
    data = meta.prepare_data(csv_files)
    
    print(f"Using best parameters from SLURM logs...")
    # These were the parameters found during the 116-fold sweep:
    # {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'n_estimators': 400}
    meta.rf.set_params(max_depth=5, max_features='sqrt', min_samples_leaf=2, n_estimators=400)
    
    print("Running evaluation (LOPO-CV) to generate plots...")
    # This will use the parameters from the loaded joblib and skip the tune_hyperparameters step
    meta.evaluate(data)
    
    print("\nVisuals regenerated successfully! Check results/meta_pr.png and results/meta_roc.png")

if __name__ == "__main__":
    replot()
