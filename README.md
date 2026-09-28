# MIAF-Net Release

This repository contains the source code and reproducibility materials for the revised MIAF-Net manuscript.

The code is organized according to the second-round revision requirements:

| Group | Folder | Purpose |
|---:|---|---|
| 1 | `01_data_split` | Filtered CMU-MOSEI sample list and fixed 8:1:1 split |
| 2 | `02_preprocessing` | Text, image, and audio preprocessing |
| 3 | `03_model` | MIAF-Net main model |
| 4 | `04_training_evaluation` | Main training, evaluation, 20-seed runs, and Table 9 reproduction |
| 5 | `05_baseline_comparison` | Baseline comparison and Table 12 reproduction |
| 6 | `06_ablation_overfitting` | Ablation and overfitting-control tables |
| 7 | `07_robustness_reproducibility` | Robustness evaluation, Table 14 support, and expected result files |

## Main Revision Points Covered

- Same independent test set is used for MIAF-Net and baseline comparison reporting.
- Validation and test ACC/Macro-F1 are both reported.
- The fixed split files are included.
- The main checkpoint-selection criterion is best validation Macro-F1.
- Training is performed for the fixed epoch schedule and is not terminated by early stopping.
- MIAF-Net parameter count is consistently reported as 155.22M.
- Hardware-dependent timing fields are kept as table outputs rather than hidden assumptions.

## Recommended Reading Order

1. Start from `01_data_split/README.md`.
2. Then read `02_preprocessing/README.md`.
3. Continue with `03_model/README.md`.
4. Use `04_training_evaluation` for the main result.
5. Use `05_baseline_comparison` for Table 12.
6. Use `06_ablation_overfitting` for Tables 7, 8, 10, and 11.
7. Use `07_robustness_reproducibility` for Table 14 and expected results.

