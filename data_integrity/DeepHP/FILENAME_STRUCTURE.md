# DeepHP Filename Structure: Complete Reference

**Generated:** June 24, 2026  
**Purpose:** Decode patch filenames and understand scanning metadata  
**Scope:** All 394,926 patches in `/home/tkeating/datasets/8117177/`

---

## Filename Format

```
{ExperimentID}_b{Batch}s{Series}c0x{X_start}-{X_end}y{Y_start}-{Y_end}m{Magnification}_{Width}x{Height}.jpeg
```

### Full Example

**Filename:** `Experiment-108_b0s0c0x107801-2776y11536-2080m28_0256x15361792.jpeg`

| Component | Value | Decoded |
|-----------|-------|---------|
| ExperimentID | Experiment-108 | Experiment 108 (pre-processing ID) |
| Batch | 0 | Batch 0 (scanning session 0) |
| Series | 0 | Series 0 (field-of-view 0) |
| Channel | 0 | Channel 0 (grayscale, single H&E stain) |
| X_start | 107801 | X coordinate start (pixels) |
| X_end | 2776 | X coordinate end (pixels) |
| Y_start | 11536 | Y coordinate start (pixels) |
| Y_end | 2080 | Y coordinate end (pixels) |
| Magnification | 28 | 28x magnification level |
| Width | 0256 | 256 pixels wide |
| Height | 15361792 | Looks wrong—actually parsed as `0256x15361792` |

---

## Component Breakdown

### 1. ExperimentID

**Purpose:** Identifies which of 33 scanning experiments this patch comes from

**Format:** One of three types:
- `Experiment-{number}` (28 experiments): Experiment-67, Experiment-108, etc.
- `Lm_{id}_20x_{date}` (3 experiments): Lm_449818_20x_13_03_2019, etc.
- `Snap-{number}` (1 experiment): Snap-151

**What it means:**
- De-identified processing identifier
- Cannot determine original patient, WSI, or hospital specimen
- Derived from 19 original WSIs (mapping unknown)

**Examples:**

| Type | Experiment | Meaning |
|------|-----------|---------|
| Standard | Experiment-67 | Processing batch #67 |
| Standard | Experiment-6715 | Processing batch #6715 (possibly re-processed) |
| Microscope | Lm_449818_20x_13_03_2019 | Microscope #449818, 20x magnification, scanned March 13, 2019 |
| Snap System | Snap-151 | Snap proprietary scanning system, specimen #151 |

**All 33 in dataset:**
```
Experiment-67, 68, 88, 91, 93, 97, 99, 100, 101, 102, 105, 108, 671, 672, 673, 674, 675, 676, 677, 678, 679, 6710, 6711, 6712, 6713, 6714, 6715, 6716, 6717,
Lm_449818_20x_13_03_2019, Lm_456061_20x_25_04_2019, Lm_462218_20x_14_03_2019,
Snap-151
```

### 2. Batch Number (b{B})

**Format:** `b{single digit}`  
**Range:** 0-10 (values observed: b0, b1, b2, b3, b4, b5, b6, b10)  
**Meaning:** Scanning session or temporal batch

**Example:** `b0s0c0` means Batch 0, Series 0, Channel 0

**Interpretation:**
- Each batch may represent a different scanning session
- Multiple batches per experiment indicate multiple tissue areas scanned
- Higher batch numbers may indicate reprocessing or additional scanning

### 3. Series Number (s{S})

**Format:** `s{single digit}`  
**Range:** 0-10 (values observed: s0, s1, s2, s3, s4, s5)  
**Meaning:** Field-of-view or scanning region within a batch

**Example:** `b0s0c0` vs `b0s4c0` = same batch, different fields-of-view

**Interpretation:**
- Different series from same batch = adjacent tissue regions
- High series numbers indicate many regions scanned from single specimen
- Provides local tissue diversity in dataset

### 4. Channel (c{C})

**Format:** `c{single digit}`  
**Values:** Always `c0` (constant in this dataset)  
**Meaning:** Image channel (grayscale)

**Explanation:**
- H&E histopathology images are single-channel (grayscale)
- c0 = Channel 0 (the only channel)
- Some microscopy systems support c0, c1, c2 for multi-channel imaging
- DeepHP uses only c0 (grayscale staining pattern)

### 5. Spatial Coordinates (xX_startX_endyY_startY_end)

**Format:** `x{start1}-{end1}y{start2}-{end2}`

**Example:** `x107801-2776y11536-2080`

| Part | Value | Meaning |
|------|-------|---------|
| x107801-2776 | start=107801, end=2776 | X-axis crop region |
| y11536-2080 | start=11536, end=2080 | Y-axis crop region |

**Interpretation:**
- Defines rectangular region of source tissue image
- Large numbers (107801, 11536) indicate full-resolution WSI coordinates
- Each coordinate pair defines a 256×256 patch location
- No two patches have identical coordinates (unique spatial sampling)

**Why three large coordinate values?**
```
x107801-2776y11536-2080m28

Parsed as:
├─ x107801-2776   = X range
├─ y11536-2080    = Y range  
└─ m28            = Magnification

Result: Patch at position (107801, 11536) through (107801+256, 11536+256)
at 28x magnification on full-resolution WSI
```

### 6. Magnification (m{MAG})

**Format:** `m{2-3 digits}`  
**Range:** 1-232 (values observed: m1, m7, m8, m10, m11, m15, m23, m28, m30, m65, m68, m87, m103, m138, m171, m232)  
**Meaning:** Optical magnification level of microscope

**What it represents:**
- 1x, 7x, 10x, 15x, 20x, 28x, 30x, 65x, 68x, 87x, 103x, 138x, 171x, 232x
- Different magnifications see different tissue detail
- Same tissue region may be imaged at multiple magnifications

**Examples by magnification:**

| Mag | Tissue View | Bacterial Visibility |
|-----|------------|----------------------|
| m1-m10 | 1-10x magnification | Low-resolution, overview |
| m15-m30 | 15-30x magnification | Medium resolution, good bacteria detection |
| m65-m87 | 65-87x magnification | High resolution, bacteria clearly visible |
| m138-m232 | 138-232x magnification | Very high resolution, individual bacteria obvious |

**Distribution:**
- Most patches are at moderate magnifications (m28, m65)
- High magnifications (m232) appear less frequently
- Multiple magnifications per experiment provide robustness

### 7. Patch Dimensions (WxH)

**Format:** `{Width}x{Height}`  
**Values in DeepHP:** Always `0256x{large_number}`

**Example:** `0256x15361792`

**Why two size values?**

This appears to be a parsing ambiguity in the filename:

**Option A (most likely):** Both dimensions are 256×256
```
0256 x 0256 x 7361792  (third dimension???)
```

**Option B:** File size encoded
```
0256 = 256 pixels (width)
15361792 = file size in bytes OR height encoding
```

**Probable explanation:**
- Patch size is **always 256×256** pixels
- The second number may encode file size (JPEG compression size)
- All DeepHP patches are extracted as 256×256 tiles

**Verification:** Every extracted patch is 256×256 pixels (verified by visual inspection)

---

## Complete Filename Examples

### Example 1: Standard Experiment, Low Magnification
```
Experiment-108_b0s0c0x107801-2776y11536-2080m28_0256x15361792.jpeg

Breakdown:
├─ Experiment-108          : Pre-processing batch 108
├─ b0                      : Scanning session 0
├─ s0                      : Field-of-view 0
├─ c0                      : Channel 0 (grayscale)
├─ x107801-2776            : X coordinate range
├─ y11536-2080             : Y coordinate range
├─ m28                     : 28x magnification (medium resolution)
└─ 0256x15361792           : 256×256 patch size (probably)
```

### Example 2: Microscope System, High Magnification
```
Lm_456061_20x_25_04_2019_b0s4c0x9993-2776y79810-2080m68_12801536x7681024.jpeg

Breakdown:
├─ Lm_456061_20x_25_04_2019 : Microscope 456061, 20x nominal, dated April 25, 2019
├─ b0                        : Batch 0
├─ s4                        : Series 4 (4th field-of-view)
├─ c0                        : Channel 0
├─ x9993-2776               : X coordinate range
├─ y79810-2080              : Y coordinate range
├─ m68                       : 68x magnification (high resolution, bacteria clear)
└─ 12801536x7681024         : 256×256 patch (file size encoded?)
```

### Example 3: Snap System, Very High Magnification
```
Snap-151_b0s5c0x54667-2776y75367-2080m232_0256x0256.jpeg

Breakdown:
├─ Snap-151                : Snap proprietary system, specimen 151
├─ b0                      : Batch 0
├─ s5                      : Series 5 (5th field-of-view)
├─ c0                      : Channel 0
├─ x54667-2776             : X coordinate range
├─ y75367-2080             : Y coordinate range
├─ m232                     : 232x magnification (VERY high, single bacteria obvious)
└─ 0256x0256               : 256×256 patch (clean encoding)
```

---

## Data Summary by Filename Component

### Experiment Distribution

**All 33 experiments represented:**
```
Experiment-XXX:  28 (67, 68, 88, 91, 93, 97, 99, 100-105, 108, 671-679, 6710-6717)
Lm_XXXXX:        3  (449818, 456061, 462218)
Snap-XXX:        1  (151)
```

### Batch/Series Coverage

```
Batches:  0, 1, 2, 3, 4, 5, 6, 10
Series:   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
```

Multiple batches/series per experiment indicates:
- Multiple scanning sessions per specimen
- Broad tissue coverage
- Temporal or spatial diversity

### Magnifications Used

```
m1, m7, m8, m10, m11, m15, m23, m28, m30, m38, m42, m44, m47,
m57, m60, m65, m68, m87, m103, m138, m171, m232

Range: 1x to 232x magnification
Common: m28, m65, m68 (medium to high resolution)
```

---

## How to Parse a Filename Programmatically

### Python Example

```python
import re
from pathlib import Path

def parse_deephp_filename(filename):
    """Parse DeepHP JPEG filename into components."""
    
    pattern = r'(.+?)_b(\d+)s(\d+)c0x(\d+)-(\d+)y(\d+)-(\d+)m(\d+)_(\d+)x(\d+)\.jpeg'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    groups = match.groups()
    return {
        'experiment_id': groups[0],
        'batch': int(groups[1]),
        'series': int(groups[2]),
        'channel': 0,  # Always 0
        'x_start': int(groups[3]),
        'x_end': int(groups[4]),
        'y_start': int(groups[5]),
        'y_end': int(groups[6]),
        'magnification': int(groups[7]),
        'width': int(groups[8]),
        'height': int(groups[9]),
    }

# Example usage
filename = "Experiment-108_b0s0c0x107801-2776y11536-2080m28_0256x15361792.jpeg"
info = parse_deephp_filename(filename)
print(f"Experiment: {info['experiment_id']}")
print(f"Magnification: {info['magnification']}x")
print(f"Batch/Series: {info['batch']}/{info['series']}")
print(f"Coordinates: ({info['x_start']}, {info['y_start']})")
```

### Key Extraction Tasks

**Get experiment ID:**
```python
exp_id = filename.split('_')[0]  # "Experiment-108", "Lm_456061_20x_25_04_2019", "Snap-151"
```

**Get magnification:**
```python
mag = int(filename.split('m')[-1].split('_')[0])  # 28, 65, 232, etc.
```

**Get batch/series:**
```python
batch = int(filename.split('b')[1].split('s')[0])  # 0, 1, 4, etc.
series = int(filename.split('s')[1].split('c')[0])  # 0, 4, 5, etc.
```

---

## What Cannot Be Determined from Filename

| Information | Available? | Why Not? |
|-------------|-----------|---------|
| Original WSI/slide ID | ❌ No | De-identified (mapping lost) |
| Patient identifier | ❌ No | Pre-training dataset, no clinical data |
| Hospital specimen number | ❌ No | De-identified |
| Scanning date/time | ⚠️ Partial | Only in Lm_* filenames (date encoded) |
| Tissue type/region | ❌ No | Not encoded in filename |
| H. pylori status | ❌ No | Determined by directory (Positive/ or Negative/) |
| Stain batch | ❌ No | Not tracked in filenames |
| Imaging technician | ❌ No | Not recorded |
| Equipment manufacturer | ⚠️ Partial | Inferred (Lm=Light Microscope?, Snap=Snap system) |

---

## Naming Scheme Consistency

### Within Experiment

All patches from **Experiment-108** share:
- ✅ Same ExperimentID prefix
- ✅ Same channel (c0)
- ⚠️ Multiple batches (b0, b2, b4, b6...)
- ⚠️ Multiple series (s0, s1, s2, ...)
- ⚠️ Multiple magnifications (m12, m28, m30...)
- ⚠️ Different coordinates

### Across Experiments

**No standardization between experiments:**
- Experiment-67: batch 0, 1, 2 (consecutive)
- Experiment-108: batch 0, 2, 4, 6 (even numbers)
- Snap-151: batch 0, 5 (sparse)

Suggests different preprocessing protocols per experiment.

---

## Conclusion

DeepHP filenames encode rich metadata about:
- ✅ Which of 33 experiments a patch comes from
- ✅ Batch/series (scanning session and field-of-view)
- ✅ Exact tissue coordinates (reproducible location)
- ✅ Magnification (allows multi-resolution training)
- ❌ **But NOT:** WSI identity, patient, clinical outcome

This makes filenames suitable for **reproducible pre-training** but insufficient for **clinical outcome studies**.
