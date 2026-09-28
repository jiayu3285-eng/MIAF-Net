# MIAF-Net Release: 04 Training and Evaluation

This folder contains the training and evaluation workflow for reproducing the main MIAF-Net results and Table 9.

The implementation follows the revised manuscript:

- 100 training epochs
- AdamW optimizer
- learning rate = `2e-4`
- weight decay = `1e-5`
- batch size = `64`
- AMP enabled by default
- checkpoint selected by the best validation Macro-F1
- no early-stopping termination
- 20 random seeds

## Files

- `train.py`
  - Trains MIAF-Net for one seed and saves the best validation Macro-F1 checkpoint.
- `evaluate.py`
  - Evaluates a saved checkpoint on validation or independent test split.
- `run_20_seeds.py`
  - Runs `train.py` across 20 seeds.
- `reproduce_table9.py`
  - Aggregates per-seed results and reports mean ± standard deviation.
- `utils/metrics.py`
  - Accuracy, Macro-F1, class-wise metrics, and summary formatting.
- `utils/seed.py`
  - Random seed control.
- `utils/logger.py`
  - CSV and JSON logging helpers.

## Default paths

The scripts assume this release layout:

```text
MIAF-Net_release/
  01_data_split/
  02_preprocessing/
  03_model/
  04_training_evaluation/
```

You can override all important paths from the command line.

## Example

Train one seed:

```bash
python train.py --raw-root D:/datasets/mosei_work/raw_extracted --image-dir D:/datasets/mosei_work/prepared/image_dataset_8_1_1 --seed 42
```

Evaluate the best checkpoint:

```bash
python evaluate.py --checkpoint outputs/seed_42/best_val_macro_f1.pt --split test --raw-root D:/datasets/mosei_work/raw_extracted --image-dir D:/datasets/mosei_work/prepared/image_dataset_8_1_1
```

Run 20 seeds:

```bash
python run_20_seeds.py --raw-root D:/datasets/mosei_work/raw_extracted --image-dir D:/datasets/mosei_work/prepared/image_dataset_8_1_1
```

Reproduce Table 9 summary:

```bash
python reproduce_table9.py --results-dir outputs
```

