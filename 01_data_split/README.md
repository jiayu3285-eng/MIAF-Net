# MIAF-Net Release: 01 Data Split

This folder contains the data-list construction and fixed split files used for the final tri-modal MIAF-Net experiments reported in the manuscript.

## What is included

- `data/metadata/sample_list_multimodal.csv`
  - Filtered CMU-MOSEI tri-modal sample list.
  - Columns: `video_id`, `sentiment_score`, `label_3`.
  - Total samples: 1958.
- `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`
  - Fixed 8:1:1 split used in the reported validation, test, ablation, and comparison experiments.
  - Columns: `video_id`, `sentiment_score`, `label_3`.
- `data/metadata/label_map.json`
  - Label mapping used by the training scripts.
- `data/metadata/dataset_statistics.json`
  - Sample counts and label distributions for the filtered dataset and fixed split.
- `scripts/build_sample_list.py`
  - Rebuilds the filtered tri-modal sample list from raw extracted CMU-MOSEI files.
- `scripts/create_fixed_split.py`
  - Recreates the stratified 8:1:1 train/validation/test split.
- `scripts/verify_split.py`
  - Checks split integrity, overlap, sample counts, and label distribution.

## Label conversion

The original sentiment score is converted to three classes as follows:

- `neg`: sentiment score <= -0.5
- `neu`: -0.5 < sentiment score < 0.5
- `pos`: sentiment score >= 0.5

For model training, the class indices are:

- `neg` -> 0
- `neu` -> 1
- `pos` -> 2

## Manuscript-consistent split

The final tri-modal experiments use the filtered multimodal subset with text, audio, and visual samples aligned by `video_id`.

| Split | Samples | pos | neu | neg |
|---|---:|---:|---:|---:|
| Train | 1566 | 736 | 732 | 98 |
| Validation | 196 | 92 | 92 | 12 |
| Test | 196 | 92 | 91 | 13 |
| Total | 1958 | 920 | 915 | 123 |

## Example usage

Rebuild the filtered multimodal list:

```bash
python scripts/build_sample_list.py --raw-root D:/datasets/mosei_work/raw_extracted --out-csv data/metadata/sample_list_multimodal.csv
```

Recreate the fixed split:

```bash
python scripts/create_fixed_split.py --input-csv data/metadata/sample_list_multimodal.csv --out-dir data/splits --seed 42
```

Verify the released split:

```bash
python scripts/verify_split.py --sample-list data/metadata/sample_list_multimodal.csv --split-dir data/splits
```

