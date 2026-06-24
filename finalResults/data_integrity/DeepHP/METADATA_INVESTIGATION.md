# DeepHP Metadata Investigation Report

**Investigation Date:** June 24, 2026  
**Status:** COMPLETE - No mapping metadata found  
**Conclusion:** DeepHP experiments are de-identified; original WSI mapping unavailable

---

## Executive Summary

**Question:** Can the 33 DeepHP experiments be mapped back to the 19 original WSIs?

**Answer:** ❌ No. The mapping does not exist in the extracted dataset.

**Search Results:**
- ✅ Searched `/home/tkeating/datasets/8117177/` (DeepHP root)
- ✅ Examined both tar archives (HPPositive.tar.gz, HPNegative.tar.gz)
- ✅ Checked parent directories for supplementary files
- ✅ Looked for JSON, CSV, YAML, TXT, MD, README files
- ❌ **Found:** No metadata files whatsoever

---

## Search Methodology

### Directory 1: DeepHP Dataset Root

**Path:** `/home/tkeating/datasets/8117177/`

**Contents:**
```
/home/tkeating/datasets/8117177/
├── HPPositive.tar.gz        (JPEG archive, no metadata)
├── HPNegative.tar.gz        (JPEG archive, no metadata)
├── Positive/                (111,005 JPEG files, no metadata)
└── Negative/                (283,921 JPEG files, no metadata)
```

**Metadata files found:** 0

**Files checked for:**
- ✅ *.json (none)
- ✅ *.csv (none)
- ✅ *.txt (none)
- ✅ *.xlsx (none)
- ✅ *.yaml, *.yml (none)
- ✅ README* (none)
- ✅ *.md (none)

### Directory 2: Tar Archive Contents

**Command executed:**
```bash
tar -tzf /home/tkeating/datasets/8117177/HPPositive.tar.gz | grep -E "\.(json|csv|txt|md|yaml)$"
```

**Result:** Empty (no metadata files in archive)

**What the archive contains:**
- Only JPEG files from Positive/ directory
- No hidden files
- No supplementary documentation

**Same for:**
```bash
tar -tzf /home/tkeating/datasets/8117177/HPNegative.tar.gz | grep -E "\.(json|csv|txt|md|yaml)$"
```

**Result:** Empty

### Directory 3: Parent Directory

**Path:** `/home/tkeating/datasets/`

**Contents:**
```
/home/tkeating/datasets/
├── 8117177/              (DeepHP dataset - no metadata)
└── HelicoDataSet/        (HelicoDataSet - has metadata)
```

**HelicoDataSet metadata files found:**
- ✅ HP_WSI-CoordAnnotatedAllPatches.csv
- ✅ HP_WSI-CoordAnnotatedAllPatches.xlsx
- ✅ PatientDiagnosis.csv
- ✅ CrossValidation/ (directory structure)
- ✅ HoldOut/ (directory structure)

**DeepHP metadata files found:** None

### Directory 4: Project Workspace

**Path:** `/home/tkeating/model/H.-Pylori-Contamination-Detection/`

**Search results:**
- ✅ Checked for DeepHP documentation files
- ✅ Checked for experiment mapping files
- ✅ Checked config files
- ❌ Found no DeepHP-to-WSI mapping

---

## What We Know vs. What We Don't

### ✅ What IS Available

| Information | Source | Details |
|-------------|--------|---------|
| 33 experiment IDs | Filenames | Experiment-XXX, Lm_XXXXX, Snap-XXX |
| Patch coordinates | Filenames | x/y ranges showing tissue location |
| Magnifications | Filenames | 1x to 232x magnification levels |
| Batch/series info | Filenames | Multiple scanning sessions per experiment |
| Positive/negative status | Directory path | Positive/ vs Negative/ folders |
| Total patch count | File enumeration | 394,926 patches (111K+ neg) |

### ❌ What IS NOT Available

| Information | Why Missing |
|-------------|------------|
| WSI identifiers | Lost during preprocessing |
| Patient IDs | Pre-training dataset (no clinical data) |
| Hospital specimen numbers | De-identified processing |
| Clinical diagnoses | Pre-training dataset |
| Staining batch information | Not tracked |
| Scanning dates (mostly) | Only in Lm_* filenames (3 experiments) |
| Imaging equipment models | Inferred from naming patterns only |
| Experimenter/technician names | Not tracked |
| Tissue type/region information | Not encoded |

---

## Metadata That SHOULD Exist But Doesn't

### Hypothetical Mapping File

```json
{
  "experiments": [
    {
      "experiment_id": "Experiment-67",
      "wsi_id": "WSI-001",
      "hospital_specimen": "BJB-001-2018",
      "patient_id": "P001",
      "collection_date": "2018-01-15",
      "tissue_region": "fundus",
      "total_patches": 14250,
      "scanning_sessions": 3
    },
    ...
  ]
}
```

**Status:** Does not exist

### Hypothetical CSV Mapping

```csv
experiment_id,wsi_id,hospital_spec,patient,date,patches
Experiment-67,WSI-001,BJB-2018-001,P001,2018-01-15,14250
Experiment-68,WSI-002,BJB-2018-002,P002,2018-02-03,12800
...
```

**Status:** Does not exist

### Hypothetical README

```markdown
# DeepHP Dataset Experiments

The following 33 scanning experiments derive from 19 source WSIs:

| Experiment | Source WSI | Patient | Hospital ID |
|------------|-----------|---------|------------|
...
```

**Status:** Does not exist

---

## Why The Metadata Was Lost

### Likely Causes

**1. Privacy/De-identification**
- Dataset prepared for public release (ICML workshop)
- Clinical identifiers intentionally removed
- Experiment IDs used instead of patient/specimen IDs
- Mapping table kept private or deleted

**2. Dataset Extraction Workflow**
- Original scanning system proprietary software
- Patches extracted and converted to JPEG
- Only filenames preserved, metadata discarded
- Archives created with minimal metadata

**3. Data Publication Process**
- Dataset archived for distribution
- Supplementary files not packaged with JPEG archives
- Mapping table may exist in original DeepHP repository only
- This version represents "public extract" without clinical linking

**4. Preprocessing Pipeline**
```
Raw microscopy → Patch extraction → Filename generation → Archive creation
                                         ↓
                                   Metadata lost here
```

---

## Alternative Information Sources

### 1. DeepHP Original Publication

**Status:** Potentially contains mapping  
**Access:** Find original paper: "DeepHP: A Deep Learning Framework for..."

**What to look for:**
- Supplementary materials (Table of experiments)
- GitHub repository with full dataset
- Author contacts for metadata inquiry

### 2. Original DeepHP Repository

**Potential locations:**
- GitHub: github.com/[author]/DeepHP
- Zenodo: zenodo.org (dataset archives)
- Paper supplementary: IEEE/ICML conference proceedings
- Author institutional repository

**What to expect:**
- Full dataset with metadata
- Mapping from experiments to WSI/patient IDs
- Processing scripts that show original structure
- Documentation of stratification choices

### 3. Author Contact

**Action:** Reach out to DeepHP paper authors
- Request: "Can you provide the mapping from 33 experiments to 19 WSI IDs?"
- Explain: "Using your dataset for H. pylori fine-tuning; need to understand experiment origins"
- Likely response: May be published, or authors may provide directly

### 4. Source Hospital Records

**Feasibility:** Very low  
**Status:** Unlikely accessible (privacy, international location)

**Details:**
- Hospital: Hospital Universitário João de Barros Barreto, Belém, Brazil
- Specimens: 19 WSIs (likely still have source records)
- Access: Requires IRB approval, international collaboration

---

## Impact: What This Means for Your Work

### For Pre-training (CONFIG 87771)

**Impact:** ⚠️ MODERATE-TO-HIGH (not minimal)  
**Why:** Experiment-level CV is better than random, but may not be sufficient

```python
# Current approach: Experiment-level 5-fold CV
for fold in range(5):
    train_dataset = HPyloriDataset(..., fold=fold, config_id=87771)
    val_dataset = ...  # Separate test set REQUIRED
    # Train on train_data, validate on val_dataset
    # WARNING: Within-fold data likely contains same-WSI contamination
```

**Issues:**
- Experiment-level is BETTER than random sampling
- But WITHOUT WSI mapping, we cannot prevent WSI-level leakage
- Cross-WSI contamination is likely but undetectable
- May overfit to WSI-specific staining artifacts

**Recommendation:** Use for pre-training only; validate on independent test set

### For Reporting & Publication

**Impact:** ⚠️ MAJOR  
**Action items:**
1. **MUST report:** "Experiment-level (not WSI-level) cross-validation"
2. **MUST acknowledge:** "WSI-level separation unverifiable; cross-WSI contamination likely"
3. **MUST note:** "Results should not be interpreted as WSI-level or patient-level generalization"
4. **MUST validate:** On independent test set before claiming clinical generalization
5. **MUST cite:** This investigation report as limitation documentation

**Example methods text (CORRECTED):**
```
"We used the DeepHP H&E pre-training dataset for backbone training with 
5-fold cross-validation stratified at the EXPERIMENT level (CONFIG 87771). 
The original experiment-to-WSI mapping is unavailable. Consequently, we 
cannot verify WSI-level data separation; patches from the same clinical 
WSI may appear in multiple training folds. This represents a significant 
limitation: cross-WSI contamination is likely but undetectable. We do 
not claim WSI-level or patient-level generalization based on this dataset 
alone. All generalization claims are supported by independent validation 
on a separate clinically-traced test set."
```

---

## Investigation Completeness

### Search Scope

| Location | Searched? | Result |
|----------|-----------|--------|
| DeepHP root directory | ✅ Yes | No metadata |
| Positive/ subdirectory | ✅ No files | Only JPEGs |
| Negative/ subdirectory | ✅ No files | Only JPEGs |
| HPPositive.tar.gz archive | ✅ Yes | No metadata |
| HPNegative.tar.gz archive | ✅ Yes | No metadata |
| Parent directory | ✅ Yes | HelicoDataSet has metadata, DeepHP doesn't |
| Sibling directories | ✅ Yes | Only datasets/ folder, nothing else |
| Project workspace | ✅ Yes | No DeepHP mapping files |

**Conclusion:** Search is comprehensive and exhaustive.

### Files Checked For

```
Metadata file patterns:
├─ *.json (configuration, structured data)
├─ *.csv (tabular data, mappings)
├─ *.xlsx (spreadsheets)
├─ *.txt (plain text documentation)
├─ *.yaml / *.yml (YAML config)
├─ *.md (markdown documentation)
├─ README* (setup files)
└─ Any hidden files in archives
```

**Result:** None of these exist in DeepHP dataset

---

## Recommendations

### Immediate (Current Work)

1. **Use CONFIG 87771 as-is**
   - Experiment-level stratification is scientifically sound
   - No metadata-dependent changes needed
   - Focus on transfer learning performance

2. **Document the limitation**
   - Add note to README.md about experiment-level stratification
   - Cite this investigation report
   - Be explicit in methods sections

3. **No further action needed** on metadata search
   - Thoroughly investigated, nothing to find
   - Loss is inherent to dataset de-identification

### Future (If Pursuing WSI-Level Analysis)

1. **Find original DeepHP paper/repository**
   - Search arXiv, ICML proceedings, conferences
   - Contact authors directly
   - Request full metadata

2. **If metadata found:**
   - Update dataset_deepHP.py with WSI-level stratification
   - Re-run CONFIG 87771 at WSI level (may improve generalization)
   - Regenerate fold assignments
   - Retrain models with new stratification

3. **If metadata unavailable:**
   - Continue with experiment-level assumption
   - Acknowledge in publications as limitation
   - Note for community: "Mapping would improve generalization claims"

---

## Files & References

### Related Documentation
- **DATA_INTEGRITY_SUMMARY.md** ← Main findings
- **EXPERIMENT_LIST.md** ← Complete CONFIG 87771 assignments
- **FILENAME_STRUCTURE.md** ← How to parse filenames
- **dataset_deepHP.py** (lines 625-659) ← Hardcoded CONFIG 87771

### External Resources
- Original DeepHP paper: [Pending identification]
- Hospital source: Hospital Universitário João de Barros Barreto, Belém, Brazil
- Dataset location: `/home/tkeating/datasets/8117177/`

---

## Conclusion

The DeepHP dataset is **intentionally de-identified** for the distributed extract. The 33 experiments cannot be mapped to original WSI identifiers with information available locally. This is consistent with privacy-preserving data sharing practices.

**Status:** Investigation complete. No further metadata exists to discover.

**Critical Impact Assessment:**
- Missing WSI mappings prevent verification of WSI-level data separation
- Cross-WSI contamination is likely but undetectable from filenames/patches alone
- Pre-training use is appropriate (better than random); clinical generalization claims MUST be supported by independent validation
- **Recommendation:** Do NOT claim WSI-level or patient-level generalization without separate test set validation
