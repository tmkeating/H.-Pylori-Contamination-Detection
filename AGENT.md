# AI Collaboration Log (AGENT.md)

## 1. Interaction Overview
- **Primary AI Tool:** Gemini 3 Flash
- **Role:** Technical Consultant & Data Strategist.

## 2. Key Contributions
- **Data Splitting Strategy:** The agent identified a "Data Leakage" flaw where patches from the same patient were present in both training and validation sets. 
- **Fix:** Implemented a strict Patient-Level Split using `folder_name.split('_')[0]` to ensure 100% isolation of independent patient biopsies.
- **Data Supplementation:** Advised on adding ~50,000 negative patches from "NEGATIVA" diagnosed patients to address the 50:1 class imbalance.
- **Model Architecture:** Suggested upscaling patches to 448x448 to leverage pre-trained ResNet18 filters for resolving fine bacterial filaments.

## 3. Human-in-the-Loop Interventions (The "Skepticism" Log)
Critical project decisions where the human developer challenged the AI:
- **Challenge 1 (Run 09):** AI reported 100% Recall. Human developer flagged this as "too good to be true" and initiated a "Stress Test."
- **Verification:** Human developer forced the AI to report performance on "Hard" (Annotated) vs "Easy" (Supplemental) tissue.
- **Result:** Accuracy dropped to a realistic 93% on hard tissue, confirming the model was no longer "cheating" and was generalized.

## 4. Known Failure Modes & Future Work
- **Stain Variation:** Identified that the model struggles with "Weak Stainer" phenotypes (e.g., patients B22-174 and B22-102).
- **Proposed Solution:** Future iterations should include **Stain Normalization (Macenko Method)** to standardize IHC contrast before inference.