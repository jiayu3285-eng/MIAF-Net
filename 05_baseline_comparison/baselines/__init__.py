"""Baseline model registry."""

from .bbfn import BBFN
from .deep_hoseq import DeepHOSeq
from .kuda import KuDA
from .lmf import LMF
from .mlmf import MLMF
from .mult import MulT
from .tfn import TFN
from .almt import ALMT


BASELINE_REGISTRY = {
    "tfn": TFN,
    "lmf": LMF,
    "mult": MulT,
    "deep_hoseq": DeepHOSeq,
    "bbfn": BBFN,
    "mlmf": MLMF,
    "almt": ALMT,
    "kuda": KuDA,
}

