# MIAF-Net Release: 06 Ablation and Overfitting Controls

This folder contains the ablation and overfitting-control code used to support Tables 7, 8, 10, and 11 in the revised manuscript.

## Covered experiments

Progressive replacement:

- `MIAF-base`
- `MIAF-T`
- `MIAF-TI`
- `MIAF-TIA`
- `MIAF-Net`

Fusion ablation:

- direct concatenation
- DCMIF only
- token attention
- cross attention
- bilinear gate
- DCMIF + MHSA

Regularization and overfitting controls:

- frozen encoders
- full fusion head
- linear concat classifier
- small concat MLP
- dropout controls
- weight decay controls
- hidden-size controls

## Files

```text
models/
  progressive_variants.py
  fusion_variants.py
experiments/
  ablation/
    reproduce_table10.py
    reproduce_table11.py
  overfitting/
    reproduce_table7.py
    reproduce_table8.py
configs/
  ablations/
  regularization/
utils/
  aggregate.py
```

The scripts aggregate result JSON files into manuscript-style tables. They are intentionally separated from the core training loop so that all variants can reuse the same split, preprocessing, metric, and checkpoint-selection protocol from earlier groups.

