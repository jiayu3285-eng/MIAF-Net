# MIAF-Net Release: 02 Preprocessing

This folder contains the preprocessing code used by the final tri-modal MIAF-Net experiments.

The implementation follows the revised manuscript:

- Text:
  - Read transcript files by `video_id`.
  - Normalize whitespace.
  - Tokenize with the RoBERTa tokenizer.
  - Truncate or pad each sequence to 128 tokens.
  - Return `input_ids` and `attention_mask`.
- Image:
  - Use three deterministic frames per video: 25%, 50%, and 75% of the selected segment.
  - For final tri-modal fusion experiments, use deterministic transforms only:
    - resize the shorter side to `int(224 * 1.15) = 257`
    - center crop to `224 x 224`
    - convert to tensor
    - normalize with ImageNet mean and standard deviation
  - Random visual augmentation is kept as an optional helper for auxiliary visual-branch experiments only.
- Audio:
  - Convert multi-channel waveforms to mono.
  - Resample to 16 kHz.
  - Clip or right-pad each waveform to 4 seconds, i.e. 64,000 samples.
  - Convert to a 64-bin log-Mel spectrogram.
  - Apply sample-wise global standardization over the whole log-Mel matrix.

## Files

- `miafnet/preprocessing.py`
  - Shared preprocessing functions for text, image, and audio.
- `scripts/extract_image_frames.py`
  - Builds the prepared image-frame dataset from segmented videos and the fixed split files.
- `scripts/check_preprocessing_shapes.py`
  - Checks the output tensor shapes for a small number of samples.

## Expected input layout

The scripts assume the following raw data layout, but all paths are configurable:

```text
raw_extracted/
  transcript_full/
  transcript_segmented/
  WAV_16000/
  videos_segmented/
```

The fixed split files from `01_data_split/data/splits` are used as input:

```text
01_data_split/data/splits/
  train.csv
  val.csv
  test.csv
```

## Example usage

Extract deterministic image frames:

```bash
python scripts/extract_image_frames.py ^
  --video-dir D:/datasets/mosei_work/raw_extracted/videos_segmented ^
  --split-dir ../01_data_split/data/splits ^
  --out-dir D:/datasets/mosei_work/prepared/image_dataset_8_1_1
```

Check preprocessing shapes:

```bash
python scripts/check_preprocessing_shapes.py ^
  --split-csv ../01_data_split/data/splits/test.csv ^
  --transcript-dir D:/datasets/mosei_work/raw_extracted/transcript_full ^
  --audio-dir D:/datasets/mosei_work/raw_extracted/WAV_16000 ^
  --image-dir D:/datasets/mosei_work/prepared/image_dataset_8_1_1 ^
  --split test
```

