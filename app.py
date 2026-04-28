from __future__ import annotations

import math
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from metric_catalog import MetricInfo, MetricMode, metric_options, metrics_for_mode


st.set_page_config(page_title="IQA Metric UI", layout="centered")


@st.cache_resource(show_spinner=False)
def get_available_pyiqa_models() -> set[str] | None:
    try:
        import pyiqa
    except ImportError:
        return None

    try:
        return {str(name).lower() for name in pyiqa.list_models()}
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_metric(metric_name: str, device: str) -> Any:
    import pyiqa

    return pyiqa.create_metric(metric_name, device=device)


def resolve_device(device_choice: str) -> str:
    import torch

    if device_choice == "CUDA":
        if not torch.cuda.is_available():
            st.warning("CUDA was selected, but PyTorch cannot see a CUDA device. Using CPU.")
            return "cpu"
        return "cuda"
    if device_choice == "CPU":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def image_to_tensor(image: Image.Image, device: str) -> Any:
    import torch

    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def score_to_float(score: Any) -> float:
    import torch

    if isinstance(score, torch.Tensor):
        values = score.detach().cpu().reshape(-1)
        if values.numel() != 1:
            raise ValueError(f"Expected one scalar score, got {values.numel()} values.")
        return float(values[0])
    if isinstance(score, np.ndarray):
        values = score.reshape(-1)
        if values.size != 1:
            raise ValueError(f"Expected one scalar score, got {values.size} values.")
        return float(values[0])
    return float(score)


def compute_score(
    metric_info: MetricInfo,
    test_image: Image.Image,
    reference_image: Image.Image | None,
    device: str,
) -> float:
    import torch

    metric = load_metric(metric_info.key, device)
    test_tensor = image_to_tensor(test_image, device)

    with torch.no_grad():
        if metric_info.mode == "FR":
            if reference_image is None:
                raise ValueError("A reference image is required for FR metrics.")
            ref_tensor = image_to_tensor(reference_image, device)
            score = metric(test_tensor, ref_tensor)
        else:
            score = metric(test_tensor)

    return score_to_float(score)


def open_image(uploaded_file: Any) -> Image.Image | None:
    if uploaded_file is None:
        return None
    return Image.open(uploaded_file).convert("RGB")


def format_score(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.6g}"


st.title("IQA Metric UI")

available_models = get_available_pyiqa_models()
if available_models is None:
    st.info("pyIQA is not available yet, or its model list could not be loaded.")

with st.sidebar:
    mode = st.radio("Metric type", ["FR", "NR"], horizontal=True)
    device_choice = st.selectbox("Device", ["Auto", "CPU", "CUDA"], index=0)

all_mode_metrics = metrics_for_mode(mode)  # type: ignore[arg-type]
visible_metrics = metric_options(mode, available_models)  # type: ignore[arg-type]
if not visible_metrics:
    visible_metrics = list(all_mode_metrics.values())

metric_by_label = {
    f"{metric.display_name} ({metric.key})": metric for metric in visible_metrics
}
metric_label = st.selectbox("Metric", list(metric_by_label.keys()))
metric_info = metric_by_label[metric_label]

info_cols = st.columns(3)
info_cols[0].metric("Type", metric_info.mode)
info_cols[1].metric("Higher better", "Yes" if metric_info.higher_is_better else "No")
info_cols[2].metric("pyIQA key", metric_info.key)
st.caption(metric_info.description)

if available_models is not None and metric_info.key not in available_models:
    st.warning(f"`{metric_info.key}` is not listed by this pyIQA installation.")

reference_image = None
test_image = None

if metric_info.mode == "FR":
    ref_upload, test_upload = st.columns(2)
    with ref_upload:
        reference_file = st.file_uploader(
            "Reference image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
            key="reference_image",
        )
        reference_image = open_image(reference_file)
        if reference_image is not None:
            st.image(reference_image, use_container_width=True)
    with test_upload:
        test_file = st.file_uploader(
            "Distorted/test image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
            key="test_image_fr",
        )
        test_image = open_image(test_file)
        if test_image is not None:
            st.image(test_image, use_container_width=True)
else:
    test_file = st.file_uploader(
        "Image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        key="test_image_nr",
    )
    test_image = open_image(test_file)
    if test_image is not None:
        st.image(test_image, use_container_width=True)

can_score = test_image is not None and (metric_info.mode == "NR" or reference_image is not None)

if st.button("Compute score", type="primary", disabled=not can_score):
    try:
        device = resolve_device(device_choice)
        with st.spinner(f"Running {metric_info.display_name} on {device}..."):
            value = compute_score(metric_info, test_image, reference_image, device)
        st.metric("Metric value", format_score(value))
        st.caption(metric_info.direction_text)
    except ImportError as exc:
        st.error(f"Missing dependency: {exc}")
    except Exception as exc:
        st.error(f"Could not compute `{metric_info.key}`: {exc}")
