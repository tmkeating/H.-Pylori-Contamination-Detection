# DeepHP CONFIG 87771: Complete Experiment List and Fold Assignments

**Generated:** June 24, 2026  
**Config ID:** 87771  
**Total Experiments:** 33  
**Cross-validation:** 5-fold, experiment-level stratification  
**Data Source:** greedy_fold_configs.json (authoritative source)

---

## Fold 0: 7 Validation Experiments

**Validation Set:** 85,854 patches | Ratio: 1:2.33

| Experiment | Type |
|---|---|
| Experiment-679 | Standard |
| Lm_449818_20x_13_03_2019 | Microscope (20x, 13-03-2019) |
| Experiment-677 | Standard |
| Experiment-88 | Standard |
| Experiment-6716 | Standard |
| Experiment-68 | Standard |
| Experiment-671 | Standard |

**Training Set:** 309,071 patches | Ratio: 1:2.27 | 26 experiments (all other experiments)

---

## Fold 1: 10 Validation Experiments

**Validation Set:** 37,093 patches | Ratio: 1:2.06

| Experiment | Type |
|---|---|
| Experiment-678 | Standard |
| Experiment-6712 | Standard |
| Experiment-673 | Standard |
| Experiment-101 | Standard |
| Experiment-6710 | Standard |
| Experiment-6717 | Standard |
| Experiment-676 | Standard |
| Experiment-672 | Standard |
| Experiment-99 | Standard |
| Experiment-6713 | Standard |

**Training Set:** 357,832 patches | Ratio: 1:2.31 | 23 experiments (all other experiments)

---

## Fold 2: 5 Validation Experiments

**Validation Set:** 78,085 patches | Ratio: 1:2.31

| Experiment | Type |
|---|---|
| Experiment-97 | Standard |
| Experiment-674 | Standard |
| Lm_456061_20x_25_04_2019 | Microscope (20x, 25-04-2019) |
| Experiment-675 | Standard |
| Experiment-91 | Standard |

**Training Set:** 316,840 patches | Ratio: 1:2.27 | 28 experiments (all other experiments)

---

## Fold 3: 4 Validation Experiments

**Validation Set:** 78,189 patches | Ratio: 1:2.29

| Experiment | Type |
|---|---|
| Experiment-6711 | Standard |
| Experiment-108 | Standard |
| Experiment-105 | Standard |
| Lm_462218_20x_14_03_2019 | Microscope (20x, 14-03-2019) |

**Training Set:** 316,736 patches | Ratio: 1:2.28 | 29 experiments (all other experiments)

---

## Fold 4: 7 Validation Experiments

**Validation Set:** 115,704 patches | Ratio: 1:2.29

| Experiment | Type |
|---|---|
| Experiment-67 | Standard |
| Experiment-100 | Standard |
| Experiment-102 | Standard |
| Experiment-93 | Standard |
| Snap-151 | Snap System |
| Experiment-6715 | Standard |
| Experiment-6714 | Standard |

**Training Set:** 279,221 patches | Ratio: 1:2.28 | 26 experiments (all other experiments)

---

## Summary: All 33 Experiments Distributed Across 5 Folds

### Experiments by Type

```
Experiment-XXX (Standard):     28 experiments
├─ Single/double digit: 67, 68, 88, 91, 93, 97, 99, 100, 101, 102, 105, 108
├─ 6xx series: 671, 672, 673, 674, 675, 676, 677, 678, 679
└─ 6xxx series: 6710, 6711, 6712, 6713, 6714, 6715, 6716, 6717

Lm_XXXXX_20x_XXXX_XXXX (Microscope):  3 experiments
├─ Lm_449818_20x_13_03_2019  (Fold 0)
├─ Lm_456061_20x_25_04_2019  (Fold 2)
└─ Lm_462218_20x_14_03_2019  (Fold 3)

Snap-XXX (Snap System):        1 experiment
└─ Snap-151                     (Fold 4)

TOTAL:                         33 experiments (all uniquely assigned)
```

### Fold Distribution at a Glance

| Fold | Val Experiments | Val Patches | Val Ratio | Train Experiments | Train Patches | Train Ratio |
|------|---|---|---|---|---|---|
| **0** | 7 | 85,854 | 1:2.33 | 26 | 309,071 | 1:2.27 |
| **1** | 10 | 37,093 | 1:2.06 | 23 | 357,832 | 1:2.31 |
| **2** | 5 | 78,085 | 1:2.31 | 28 | 316,840 | 1:2.27 |
| **3** | 4 | 78,189 | 1:2.29 | 29 | 316,736 | 1:2.28 |
| **4** | 7 | 115,704 | 1:2.29 | 26 | 279,221 | 1:2.28 |
| **TOTAL** | **33** | **394,925** | **1:2.28** | **26-29 per fold** | **1,579,700** | **1:2.28** |

**Key Observation:** All training sets maintain excellent class balance (~1:2.27-2.31), while validation ratios vary slightly by fold (1:2.06-2.33).

---

## Key Observations

### Validation Set Characteristics

- **Fold 0:** 85,854 patches (21.8% of total) from 7 experiments
- **Fold 1:** 37,093 patches (9.4% of total) from 10 experiments  
- **Fold 2:** 78,085 patches (19.8% of total) from 5 experiments
- **Fold 3:** 78,189 patches (19.8% of total) from 4 experiments
- **Fold 4:** 115,704 patches (29.3% of total) from 7 experiments

**Total validation patches across all folds:** 394,925 (sum of all 5 folds)

### Training Set Characteristics

- **All folds maintain excellent class balance:** 1:2.27-2.31 negative:positive ratio
- **Training set sizes vary:** 26-29 experiments per fold
- **Total training patches across all folds:** 1,579,700 (309K+357K+316K+316K+279K)

### Experiment Numbering Patterns

The 28 standard experiments follow naming conventions suggesting processing order/batches:
- **Single/double digit (67-108):** Early processing (12 experiments)
- **600-series (671-679):** Sequential grouping (9 experiments)
- **6000-series (6710-6717):** Likely re-processing or separate batch (8 experiments)
- **Microscope IDs (Lm_*):** 20x magnification with date stamps (3 experiments)
- **Snap system:** Single proprietary scanner (1 experiment)

---

## Cross-fold Contamination: NONE ✓

**Verification:**
- Each of 33 experiments appears in exactly one fold (no overlap)
- Zero patches assigned to multiple folds
- Complete partition: Folds 0-4 contain all 33 experiments
- Validation sets sum to exactly 394,925 patches

---

## Recommendations for Use

### Training with CONFIG 87771

```python
from dataset_deepHP import HPyloriDataset
from config import DEEPHP_DATASET_ROOT

# Load fold-specific dataset
for fold in range(5):
    val_dataset = HPyloriDataset(
        root=DEEPHP_DATASET_ROOT,
        fold=fold,
        config_id=87771,  # Uses CONFIG 87771 experiment assignments
        split='val'
    )
    train_dataset = HPyloriDataset(
        root=DEEPHP_DATASET_ROOT,
        fold=fold,
        config_id=87771,
        split='train'
    )
    print(f"Fold {fold}: val={len(val_dataset)}, train={len(train_dataset)}")

# Expected output:
# Fold 0: val=85854, train=309071
# Fold 1: val=37093, train=357832
# Fold 2: val=78085, train=316840
# Fold 3: val=78189, train=316736
# Fold 4: val=115704, train=279221
```

### Cross-Validation Loop

```bash
# All folds guarantee:
# - No experiment appears in multiple folds
# - No patch contamination across folds
# - Diverse scanning conditions in each fold
# - Experiment-level stratification (not WSI-level; see limitations)
# - Total dataset: 394,925 validation patches
```

### Reporting in Publication

**Recommended wording for methods section:**

"We evaluated the DeepHP pre-trained backbone using CONFIG 87771, a 5-fold cross-validation strategy with experiment-level stratification. The dataset comprises 33 scanning experiments from 19 clinical whole-slide images, collectively containing 394,925 patches (120,374 positive [H. pylori], 274,551 negative; imbalance ratio 1:2.28). Each of the 33 experiments was assigned to exactly one fold: Fold 0 (7 experiments, 85,854 patches, 1:2.33), Fold 1 (10 experiments, 37,093 patches, 1:2.06), Fold 2 (5 experiments, 78,085 patches, 1:2.31), Fold 3 (4 experiments, 78,189 patches, 1:2.29), Fold 4 (7 experiments, 115,704 patches, 1:2.29). All training sets maintained excellent class balance (1:2.27-2.31 ratio).

**Critical limitation:** This dataset has no accessible experiment-to-WSI mapping (de-identified during preprocessing), preventing true WSI-level separation verification. Experiments represent scanning sessions, not original clinical specimens. Consequently, cross-WSI contamination at the patch level is possible and likely but undetectable with current de-identified data. Empirically, Folds 1 and 4 showed anomalously high validation accuracy, suggesting likely cross-WSI patch contamination. Therefore, claims of WSI-level or patient-level generalization cannot be supported by this 5-fold CV alone; independent validation on a clinically-traced test set is required."

---

## Files Referenced

- **dataset_deepHP.py** (lines 625-659): Hardcoded CONFIG 87771 experiment assignments
- **DATA_INTEGRITY_SUMMARY.md**: Overall dataset quality and limitations
- **FILENAME_STRUCTURE.md**: Detailed explanation of JPEG filename components
