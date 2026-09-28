# MIAF-Net Release: 07 Robustness and Reproducibility Materials

This folder contains robustness evaluation and auxiliary reproduction materials for the revised manuscript.

## Purpose

This group supports Table 14 and the supplementary reproducibility notes.

It includes:

- perturbation generation
- modality-missing simulation
- robustness evaluation
- result aggregation
- expected result files for quick comparison

## Files

```text
experiments/
  robustness/
    perturbations.py
    reproduce_table14.py
results/
  expected/
    table9_results.csv
    table12_results.csv
    table14_results.csv
README.md
requirements.txt
```

## Robustness settings

The supported perturbation settings are:

- `clean`
- `visual_gaussian_noise`
- `visual_center_occlusion`
- `audio_gaussian_noise`
- `visual_missing`
- `audio_missing`
- `text_missing`
- `image_audio_missing`

For each setting, the script reports:

- ACC
- Macro-F1
- ACC drop relative to the clean test setting
- Macro-F1 drop relative to the clean test setting

## Example

```bash
python experiments/robustness/reproduce_table14.py ^
  --checkpoint ../04_training_evaluation/outputs/seed_42/best_val_macro_f1.pt ^
  --raw-root D:/datasets/mosei_work/raw_extracted ^
  --image-dir D:/datasets/mosei_work/prepared/image_dataset_8_1_1 ^
  --out-csv results/table14_reproduced.csv
```

## Expected results

The `results/expected` folder contains manuscript-facing expected tables. `table9_results.csv` and `table12_results.csv` reflect the revised main and baseline-comparison results. `table14_results.csv` provides the expected schema for robustness results and should be updated with the final robustness runs if new values are generated.

