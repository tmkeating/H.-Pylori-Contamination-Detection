# DeepHP Data Integrity Documentation

**Location:** `/home/tkeating/model/H.-Pylori-Contamination-Detection/finalResults/data_integrity/DeepHP/`

**Purpose:** Documentation of the DeepHP H&E pre-training dataset (394,925 patches, 33 experiments, 5-fold CV)

---

## Quick Reference

| Metric | Value |
|--------|-------|
| **Total Patches** | 394,926 |
| **Positive (H. pylori)** | 120,375 (30.5%) |
| **Negative (background)** | 274,551 (69.5%) |
| **Source WSIs** | 19 clinical slides |
| **Derived Experiments** | 33 scanning experiments |
| **Config ID** | 87771 (experiment-level 5-fold CV) |
| **Cross-fold Contamination** | NONE ✓ |
| **WSI Mapping Available** | NO ✗ |

---

## Documentation Files

| File | Purpose | Use Case |
|------|---------|----------|
| **DATA_INTEGRITY_SUMMARY.md** | Main report with composition, quality, stratification | Understanding dataset & limitations |
| **EXPERIMENT_LIST.md** | All 33 experiments + fold assignments | Implementing 5-fold CV with CONFIG 87771 |
| **FILENAME_STRUCTURE.md** | JPEG filename format breakdown & parsing | Processing patches programmatically |
| **METADATA_INVESTIGATION.md** | Search results for experiment-to-WSI mapping | Understanding why metadata unavailable |
| **experiment_distribution.json** | Machine-readable fold statistics | Programmatic access to dataset info |

---

## Key Points

### ✅ What's Verified

- **33 experiments identified:** 28 Experiment-XXX, 3 Lm_*, 1 Snap-151
- **Experiment-level stratification sound:** All experiments in exactly one fold
- **22 magnifications available:** 1x-232x across multiple batches/series
- **Complete inventory:** All 394,926 patches accounted for, no corruption

### ⚠️ Critical Limitation: Cannot Map Experiments → WSIs

Without the 33→19 experiment-to-WSI mapping:
- **Cannot verify WSI-level separation** (same WSI likely split across folds)
- **Cross-WSI contamination likely but undetectable**
- **Cannot claim WSI/patient-level generalization** without independent test set
- **Pre-training may overfit to WSI-specific artifacts**

**How to use:**
- ✅ For pre-training/transfer learning backbones
- ✅ For feature extraction
- ❌ For standalone clinical claims (validate separately)

---

## Usage Quick Start

**5-Fold Cross-Validation:**
```python
from dataset_deepHP import HPyloriDataset
dataset = HPyloriDataset(..., fold=0, config_id=87771)
# Experiments in Fold 0: See EXPERIMENT_LIST.md
```

**Parse Filenames:**
```python
import re
pattern = r'(.+?)_b(\d+)s(\d+)c0x(\d+)-(\d+)y(\d+)-(\d+)m(\d+)_(\d+)x(\d+)\.jpeg'
exp_id, batch, series, *coords, mag = re.match(pattern, filename).groups()
```

**Report Format:**
Always state: "Experiment-level (NOT WSI-level) 5-fold CV per CONFIG 87771"  
Acknowledge: "WSI mapping unavailable; cross-WSI contamination likely but undetectable"  
Validate: "Independent test set (e.g., HelicoDataSet) used for generalization claims"

---

## Important Notes

**Do NOT:**
- Claim WSI-level or patient-level generalization from this dataset alone
- Assume experiment-level independence equals WSI-level independence  
- Expect clinical metadata (patient IDs, diagnoses, outcomes)

**DO:**
- Use for pre-training with appropriate caveats
- Report stratification level accurately
- Validate on independent test set before clinical claims
- Note limitation in publications
- Contact DeepHP authors if WSI mapping needed

---

## Usage Scenarios

### Scenario 1: Implementing 5-Fold Cross-Validation

**Files needed:** EXPERIMENT_LIST.md

**Process:**
1. Read experiment assignments from EXPERIMENT_LIST.md
2. Implement fold logic in your training code
3. Or use existing `dataset_deepHP.py` which has CONFIG 87771 hardcoded

```python
from dataset_deepHP import HPyloriDataset

for fold_idx in range(5):
    train_data = HPyloriDataset(..., fold=fold_idx, config_id=87771)
    val_data = ...  # Held-out test set from HelicoDataSet
    # Train on train_data, validate on val_data
```

---

### Scenario 2: Understanding Patch Metadata

**Files needed:** FILENAME_STRUCTURE.md

**Process:**
1. Extract filename from path: `Experiment-108_b0s0c0x107801-2776y11536-2080m28_0256x15361792.jpeg`
2. Parse using regex pattern from FILENAME_STRUCTURE.md
3. Access experiment, magnification, coordinates, etc.

```python
import re

pattern = r'(.+?)_b(\d+)s(\d+)c0x(\d+)-(\d+)y(\d+)-(\d+)m(\d+)_(\d+)x(\d+)\.jpeg'
match = re.match(pattern, filename)

exp_id = match.group(1)  # Experiment-108
magnification = int(match.group(8))  # 28
```

---

### Scenario 3: Reporting Dataset Composition

**Files needed:** DATA_INTEGRITY_SUMMARY.md, EXPERIMENT_LIST.md

**Process:**
1. Copy dataset statistics from DATA_INTEGRITY_SUMMARY.md
2. Reference CONFIG 87771 fold assignments from EXPERIMENT_LIST.md
3. MUST note that WSI-level mapping is unavailable and cross-WSI contamination is likely
4. Report as "experiment-level CV" not "WSI-level CV"

**Example methods section (CORRECTED):**
```
"The DeepHP H&E pre-training dataset comprises 394,926 patches from 
33 scanning experiments derived from 19 whole-slide images. We performed 
5-fold cross-validation at the EXPERIMENT level (CONFIG 87771), NOT at 
the WSI level. The original experiment-to-WSI mapping is unavailable; 
therefore, we cannot verify that patches from the same WSI were not split 
across train/test folds. This represents a significant limitation: cross-WSI 
contamination is likely but undetectable. Consequently, we do not make 
claims about WSI-level or patient-level generalization based on this 
dataset alone; validation on an independent test set is required."
```

---

### Scenario 4: Investigating Missing WSI Mapping

**Files needed:** METADATA_INVESTIGATION.md

**Process:**
1. Understand that no metadata exists locally
2. Contact DeepHP authors or original paper repository
3. Or accept experiment-level stratification as limitation

**Next steps if WSI mapping found:**
```
1. Create new CONFIG with WSI-level stratification
2. Group experiments by their source WSI
3. Retrain models with improved cross-validation
4. Claim WSI-level generalization (instead of experiment-level)
```

---

## Important Notes

### 🔴 Do NOT Assume

- ❌ Do NOT assume cross-WSI separation (patches from same WSI likely in multiple folds)
- ❌ Do NOT claim WSI-level generalization without independent test set
- ❌ Do NOT claim patient-level generalization based on pre-training splits alone
- ❌ Do NOT assume experiment-level independence equals WSI-level independence
- ❌ Do NOT expect to find clinical metadata in DeepHP patches

### 🟢 DO Assume

- ✅ DO assume experiment-level separation (enforced and verified)
- ✅ DO assume multi-magnification robustness (22 magnifications)
- ✅ DO assume no cross-fold experiment contamination (verified)
- ✅ DO assume de-identification is intentional (privacy)
- ✅ DO plan for independent validation on clinically-traced dataset

### 🟡 Recommend (CRITICAL)

- ⚠️ REPORT stratification as "experiment-level" in publications, NOT "WSI-level"
- ⚠️ ACKNOWLEDGE that cross-WSI contamination is likely and undetectable
- ⚠️ NOTE that WSI-level or patient-level generalization cannot be verified with this dataset alone
- ⚠️ REQUIRE independent test set (e.g., HelicoDataSet) for clinical claims
- ⚠️ CONSIDER reaching out to DeepHP authors if WSI mapping needed for improved stratification
- ⚠️ CAUTION: Pre-training may overfit to WSI-specific artifacts that won't generalize clinically

---

## Related Files & References

### In This Project

- **dataset_deepHP.py** (lines 625-659)
  - Contains hardcoded CONFIG 87771 experiment assignments
  - Implements loading logic for 5-fold cross-validation

- **README.md**
  - Project overview and execution instructions
  - References DeepHP for pre-training

- **TRANSFER_LEARNING.md**
  - Phase 1 describes DeepHP usage in pipeline
  - Explains why pre-training is beneficial

### External References

- **Original DeepHP Paper:** [To be identified]
  - Contains full dataset description
  - Likely has mapping from experiments to WSIs
  - Citation needed for publications

- **DeepHP Dataset Location:** `/home/tkeating/datasets/8117177/`
  - Positive/: 120,375 patches (30.5%)
  - Negative/: 274,551 patches (69.5%)

- **Hospital Source:** Hospital Universitário João de Barros Barreto, Belém, Brazil
  - 19 original clinical whole-slide images
  - Patient data not available in extracted dataset

---

## Version History

| Date | Changes |
|------|---------|
| 2026-06-24 | Initial investigation complete; all documentation files created |

---

## Questions?

**If you need to:**
- Understand fold assignments → Read **EXPERIMENT_LIST.md**
- Parse patch filenames → Read **FILENAME_STRUCTURE.md**
- Understand limitations → Read **METADATA_INVESTIGATION.md**
- Get dataset statistics → Read **DATA_INTEGRITY_SUMMARY.md**
- Access programmatically → Use **experiment_distribution.json**

**If you need to:**
- Add WSI-level stratification → Contact DeepHP authors for mapping
- Claim patient-level generalization → Use HelicoDataSet validation
- Verify clinical outcomes → Not possible (pre-training dataset)

---

**Investigation Status:** COMPLETE  
**Conclusion:** Dataset is suitable for transfer learning with experiment-level cross-validation. WSI-level mapping unavailable; de-identification appears intentional.
