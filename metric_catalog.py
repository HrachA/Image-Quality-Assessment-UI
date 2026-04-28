from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MetricMode = Literal["FR", "NR"]


@dataclass(frozen=True)
class MetricInfo:
    key: str
    display_name: str
    mode: MetricMode
    higher_is_better: bool
    description: str

    @property
    def direction_text(self) -> str:
        return "higher is better" if self.higher_is_better else "lower is better"


FR_METRICS: dict[str, MetricInfo] = {
    "gmsd": MetricInfo(
        "gmsd",
        "GMSD",
        "FR",
        False,
        "Gradient magnitude similarity deviation for comparing distorted and reference images.",
    ),
    "dists": MetricInfo(
        "dists",
        "DISTS",
        "FR",
        False,
        "Deep image structure and texture similarity distance.",
    ),
    "lpips": MetricInfo(
        "lpips",
        "LPIPS",
        "FR",
        False,
        "Learned perceptual image patch similarity distance.",
    ),
    "ms_ssim": MetricInfo(
        "ms_ssim",
        "MS-SSIM",
        "FR",
        True,
        "Multi-scale structural similarity between distorted and reference images.",
    ),
    "fsim": MetricInfo(
        "fsim",
        "FSIM",
        "FR",
        True,
        "Feature similarity index based on phase congruency and gradient information.",
    ),
    "ssim": MetricInfo(
        "ssim",
        "SSIM",
        "FR",
        True,
        "Structural similarity between distorted and reference images.",
    ),
    "vif": MetricInfo(
        "vif",
        "VIF",
        "FR",
        True,
        "Visual information fidelity score against a reference image.",
    ),
    "psnr": MetricInfo(
        "psnr",
        "PSNR",
        "FR",
        True,
        "Peak signal-to-noise ratio against a reference image.",
    ),
    "topiq_fr": MetricInfo(
        "topiq_fr",
        "TOPIQ-FR",
        "FR",
        True,
        "Transformer-oriented full-reference quality prediction.",
    ),
    "ahiq": MetricInfo(
        "ahiq",
        "AHIQ",
        "FR",
        True,
        "Attention-based full-reference perceptual image quality metric.",
    ),
    "ckdn": MetricInfo(
        "ckdn",
        "CKDN",
        "FR",
        True,
        "Deep full-reference image quality assessment metric.",
    ),
    "pieapp": MetricInfo(
        "pieapp",
        "PieAPP",
        "FR",
        False,
        "Learned perceptual full-reference error metric.",
    ),
}


NR_METRICS: dict[str, MetricInfo] = {
    "brisque": MetricInfo(
        "brisque",
        "BRISQUE",
        "NR",
        False,
        "Natural scene statistics based blind image quality score.",
    ),
    "niqe": MetricInfo(
        "niqe",
        "NIQE",
        "NR",
        False,
        "Natural image quality evaluator without a reference image.",
    ),
    "ilniqe": MetricInfo(
        "ilniqe",
        "IL-NIQE",
        "NR",
        False,
        "Integrated local natural image quality evaluator.",
    ),
    "piqe": MetricInfo(
        "piqe",
        "PIQE",
        "NR",
        False,
        "Perception-based image quality evaluator.",
    ),
    "nrqm": MetricInfo(
        "nrqm",
        "NRQM",
        "NR",
        True,
        "No-reference quality metric designed around restored images.",
    ),
    "nima": MetricInfo(
        "nima",
        "NIMA",
        "NR",
        True,
        "Neural image assessment model that predicts aesthetic or technical quality.",
    ),
    "cnniqa": MetricInfo(
        "cnniqa",
        "CNNIQA",
        "NR",
        True,
        "CNN-based no-reference image quality assessment model.",
    ),
    "dbcnn": MetricInfo(
        "dbcnn",
        "DBCNN",
        "NR",
        True,
        "Deep bilinear CNN no-reference image quality model.",
    ),
    "hyperiqa": MetricInfo(
        "hyperiqa",
        "HyperIQA",
        "NR",
        True,
        "Hypernetwork-based no-reference perceptual quality model.",
    ),
    "musiq": MetricInfo(
        "musiq",
        "MUSIQ",
        "NR",
        True,
        "Multi-scale image quality transformer for no-reference scoring.",
    ),
    "maniqa": MetricInfo(
        "maniqa",
        "MANIQA",
        "NR",
        True,
        "Multi-dimension attention network for no-reference image quality assessment.",
    ),
    "clipiqa": MetricInfo(
        "clipiqa",
        "CLIP-IQA",
        "NR",
        True,
        "CLIP-based no-reference perceptual image quality score.",
    ),
    "topiq_nr": MetricInfo(
        "topiq_nr",
        "TOPIQ-NR",
        "NR",
        True,
        "Transformer-oriented no-reference quality prediction.",
    ),
}


def metrics_for_mode(mode: MetricMode) -> dict[str, MetricInfo]:
    if mode == "FR":
        return FR_METRICS
    return NR_METRICS


def metric_options(mode: MetricMode, available_models: set[str] | None = None) -> list[MetricInfo]:
    metrics = list(metrics_for_mode(mode).values())
    if available_models is None:
        return metrics
    return [metric for metric in metrics if metric.key in available_models]

