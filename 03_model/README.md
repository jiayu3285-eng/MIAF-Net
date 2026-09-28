# MIAF-Net Release: 03 Main Model

This folder contains the main MIAF-Net model implementation used by the revised manuscript.

The implementation follows the manuscript-level architecture:

- Text branch:
  - RoBERTa encoder
  - BiLSTM sequence modeling
  - Additive attention pooling
- Visual branch:
  - Swin-T backbone
  - ECA channel recalibration
  - Mean aggregation over three deterministic video frames
- Audio branch:
  - Audio Mamba encoder
  - Mean pooling over the temporal hidden sequence
- Fusion branch:
  - Dual-Stage Cross-Modal Interaction Fusion (DCMIF)
  - Pairwise interaction tokens:
    - text-image multiplication
    - text-image absolute difference
    - text-audio multiplication
    - text-audio absolute difference
    - image-audio multiplication
    - image-audio absolute difference
  - Multi-Head Self-Attention (MHSA) over the six interaction tokens
  - MLP classifier for three-class sentiment prediction

## Files

- `miafnet/model.py`
  - Main MIAF-Net model.
- `miafnet/audio_mamba.py`
  - Audio Mamba module used by the audio branch.
- `scripts/check_model_shapes.py`
  - Lightweight interface check using synthetic tensors.

## Expected inputs

The model expects the preprocessed tensors from `02_preprocessing`:

```text
input_ids:      [B, 128]
attention_mask: [B, 128]
images:         [B, 3, 3, 224, 224]
mel:            [B, 64, T]
```

The output is:

```text
logits: [B, 3]
```

## Notes

- This group only defines the model architecture.
- Dataset construction and preprocessing are provided in groups `01_data_split` and `02_preprocessing`.
- Training, validation, testing, and metric logging will be organized separately in the next group.

