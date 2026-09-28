"""Manuscript-consistent preprocessing utilities for MIAF-Net."""

from __future__ import annotations

import os
import re
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from PIL import Image
from torchvision import transforms


LABELS = ["neg", "neu", "pos"]
LABEL2ID = {"neg": 0, "neu": 1, "pos": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

TEXT_BACKBONE = "roberta-base"
MAX_TEXT_LENGTH = 128

IMAGE_SIZE = 224
IMAGE_RESIZE_SHORTER_SIDE = int(IMAGE_SIZE * 1.15)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
FRAME_TAGS = ["t25", "t50", "t75"]

TARGET_SAMPLE_RATE = 16000
CLIP_SECONDS = 4.0
CLIP_SAMPLES = int(TARGET_SAMPLE_RATE * CLIP_SECONDS)
N_MELS = 64
MEL_N_FFT = 400
MEL_HOP_LENGTH = 200
EPS = 1e-6


def normalize_label(value) -> int:
    """Normalize label values to neg=0, neu=1, pos=2."""
    if isinstance(value, str):
        label = value.strip().lower()
        if label in LABEL2ID:
            return LABEL2ID[label]
        if label in {"negative"}:
            return LABEL2ID["neg"]
        if label in {"neutral"}:
            return LABEL2ID["neu"]
        if label in {"positive"}:
            return LABEL2ID["pos"]
        if re.fullmatch(r"-?\d+", label):
            value = int(label)
        else:
            raise ValueError(f"Unknown label: {value}")

    value = int(value)
    if value in {0, 1, 2}:
        return value
    if value in {-1}:
        return 0
    raise ValueError(f"Unknown label id: {value}")


def read_transcript(transcript_dir: str | os.PathLike[str], video_id: str) -> str:
    """Read a transcript and normalize whitespace."""
    root = Path(transcript_dir)
    video_id = str(video_id).strip()
    for ext in [".txt", ".transcript", ".vtt", ".srt"]:
        path = root / f"{video_id}{ext}"
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            return re.sub(r"\s+", " ", text).strip()
    return ""


def encode_text(tokenizer, text: str, max_length: int = MAX_TEXT_LENGTH) -> dict[str, torch.Tensor]:
    """Tokenize text with fixed truncation, padding, and attention mask construction."""
    encoded = tokenizer(
        str(text),
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
    }


def make_image_transform() -> transforms.Compose:
    """Deterministic transform used by final tri-modal fusion experiments."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_RESIZE_SHORTER_SIDE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )


def make_auxiliary_train_image_transform() -> transforms.Compose:
    """Optional training-only transform for auxiliary visual-branch experiments."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.00)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
        ]
    )


def build_image_index(image_dataset_dir: str | os.PathLike[str], frame_tags: list[str] | None = None) -> dict[tuple[str, str, str], str]:
    """Index prepared image frames by (split, video_id, frame_tag)."""
    frame_tags = frame_tags or FRAME_TAGS
    root = Path(image_dataset_dir)
    index: dict[tuple[str, str, str], str] = {}

    for split in ["train", "val", "test"]:
        for label in LABELS:
            label_dir = root / split / label
            if not label_dir.is_dir():
                continue
            for tag in frame_tags:
                suffix = f"_{tag}.jpg"
                for path in label_dir.glob(f"*{suffix}"):
                    video_id = path.name[: -len(suffix)]
                    index[(split, video_id, tag)] = str(path)
    return index


def load_three_frames(
    image_index: dict[tuple[str, str, str], str],
    split: str,
    video_id: str,
    transform=None,
    frame_tags: list[str] | None = None,
) -> torch.Tensor:
    """Load the deterministic three-frame visual representation."""
    transform = transform or make_image_transform()
    frame_tags = frame_tags or FRAME_TAGS
    frames = []

    for tag in frame_tags:
        path = image_index.get((split, str(video_id), tag))
        if path and os.path.exists(path):
            image = Image.open(path).convert("RGB")
        else:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        frames.append(transform(image))

    return torch.stack(frames, dim=0)


def load_fixed_waveform(
    wav_path: str | os.PathLike[str],
    target_sr: int = TARGET_SAMPLE_RATE,
    clip_seconds: float = CLIP_SECONDS,
) -> torch.Tensor:
    """Load, mono-convert, resample, and pad/crop a waveform."""
    clip_len = int(target_sr * clip_seconds)
    wav, sr = sf.read(str(wav_path))
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)

    waveform = torch.tensor(wav, dtype=torch.float32)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    if waveform.numel() < clip_len:
        waveform = torch.nn.functional.pad(waveform, (0, clip_len - waveform.numel()))
    else:
        waveform = waveform[:clip_len]
    return waveform


def waveform_to_log_mel(
    waveform: torch.Tensor,
    target_sr: int = TARGET_SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = MEL_N_FFT,
    hop_length: int = MEL_HOP_LENGTH,
    eps: float = EPS,
) -> torch.Tensor:
    """Convert waveform to sample-wise globally standardized log-Mel spectrogram."""
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )(waveform)
    mel = torch.log(mel + eps)
    mel = (mel - mel.mean()) / (mel.std() + eps)
    return mel


def load_log_mel(
    audio_dir: str | os.PathLike[str],
    video_id: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    n_mels: int = N_MELS,
    clip_seconds: float = CLIP_SECONDS,
    eps: float = EPS,
) -> torch.Tensor:
    """Load a waveform by video_id and return a 64-bin log-Mel spectrogram."""
    wav_path = Path(audio_dir) / f"{video_id}.wav"
    if not wav_path.exists():
        clip_len = int(target_sr * clip_seconds)
        mel_time = 1 + clip_len // MEL_HOP_LENGTH
        return torch.zeros(n_mels, mel_time, dtype=torch.float32)
    waveform = load_fixed_waveform(wav_path, target_sr=target_sr, clip_seconds=clip_seconds)
    return waveform_to_log_mel(waveform, target_sr=target_sr, n_mels=n_mels, eps=eps)
