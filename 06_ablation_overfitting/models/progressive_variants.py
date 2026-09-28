"""Progressive replacement variants for MIAF-Net.

These wrappers define which modality branches and fusion setting are active for
the Table 10 progressive replacement experiment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressiveVariant:
    name: str
    use_text_branch: bool
    use_image_branch: bool
    use_audio_branch: bool
    fusion_type: str
    description: str


PROGRESSIVE_VARIANTS = {
    "miaf_base": ProgressiveVariant(
        name="MIAF-base",
        use_text_branch=True,
        use_image_branch=False,
        use_audio_branch=False,
        fusion_type="text_only",
        description="Base text-only setting.",
    ),
    "miaf_t": ProgressiveVariant(
        name="MIAF-T",
        use_text_branch=True,
        use_image_branch=False,
        use_audio_branch=False,
        fusion_type="text_branch",
        description="Text branch with RoBERTa-BiLSTM-attention representation.",
    ),
    "miaf_ti": ProgressiveVariant(
        name="MIAF-TI",
        use_text_branch=True,
        use_image_branch=True,
        use_audio_branch=False,
        fusion_type="text_image",
        description="Text and image branches.",
    ),
    "miaf_tia": ProgressiveVariant(
        name="MIAF-TIA",
        use_text_branch=True,
        use_image_branch=True,
        use_audio_branch=True,
        fusion_type="tri_modal_concat",
        description="Tri-modal branches with direct fusion.",
    ),
    "miaf_net": ProgressiveVariant(
        name="MIAF-Net",
        use_text_branch=True,
        use_image_branch=True,
        use_audio_branch=True,
        fusion_type="dcmif_mhsa",
        description="Complete MIAF-Net with DCMIF and MHSA.",
    ),
}

