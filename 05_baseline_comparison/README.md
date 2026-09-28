# MIAF-Net Release: 05 Baseline Comparison

This folder contains the baseline-comparison code used to respond to Reviewer 1 Comment 1 and Comment 3.

The comparison protocol follows the revised manuscript:

- All methods use the same fixed split from `01_data_split/data/splits`.
- Both validation and independent test performance are reported.
- Metrics include:
  - validation ACC
  - validation Macro-F1
  - test ACC
  - test Macro-F1
- Model parameter scale and training time are reported with the same table format as Table 12.

## Baselines

The following baseline model wrappers are provided:

- Tensor Fusion Network (TFN)
- Low-rank Multimodal Fusion (LMF)
- Multimodal Transformer (MulT)
- Deep Higher Order Sequence Fusion (Deep-HOSeq)
- Bi-Bimodal Fusion Network (BBFN)
- Multitask Low-rank Multimodal Fusion (MLMF)
- Adaptive Language-guided Multimodal Transformer (ALMT)
- Knowledge-guided Dynamic Modality Attention (KuDA)

## Files

```text
baselines/
  common.py
  tfn.py
  lmf.py
  mult.py
  deep_hoseq.py
  bbfn.py
  mlmf.py
  almt.py
  kuda.py
configs/baselines/
  tfn.yaml
  lmf.yaml
  mult.yaml
  deep_hoseq.yaml
  bbfn.yaml
  mlmf.yaml
  almt.yaml
  kuda.yaml
extract_feature_cache.py
run_baseline.py
reproduce_table12.py
```

## Input to `reproduce_table12.py`

The script aggregates per-method result JSON files. Each JSON file should contain:

```json
{
  "model": "TFN",
  "val_acc": [0.462, "..."],
  "val_macro_f1": [0.391, "..."],
  "test_acc": [0.442, "..."],
  "test_macro_f1": [0.372, "..."],
  "params_m": 4.5,
  "training_time_min_per_run": 74
}
```

The lists should contain one value per seed. The script reports mean ± standard deviation.

## Example

Export aligned modality features:

```bash
python extract_feature_cache.py --checkpoint ../04_training_evaluation/outputs/seed_42/best_val_macro_f1.pt --raw-root D:/datasets/mosei_work/raw_extracted --image-dir D:/datasets/mosei_work/prepared/image_dataset_8_1_1 --out-dir results/features
```

Train one baseline:

```bash
python run_baseline.py --config configs/baselines/tfn.yaml --feature-dir results/features --out-json results/baselines/tfn.json
```

Aggregate Table 12:

```bash
python reproduce_table12.py --results-dir results/baselines --out-csv table12_reproduced.csv
```
