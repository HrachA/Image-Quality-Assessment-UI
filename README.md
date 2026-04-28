# IQA UI

A small Streamlit interface for running pyIQA image quality metrics.

The UI lets the user choose:

- `FR` full-reference metrics, then upload a reference image and a distorted/test image.
- `NR` no-reference metrics, then upload one image.

Each metric shows its type, pyIQA key, and whether higher values are better.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you need a specific PyTorch build for CUDA, install `torch` and `torchvision`
from the matching PyTorch index before installing the rest of the requirements.

## Run

```powershell
python app.py
```

This launches Streamlit internally, so you do not need the `streamlit` command
to be on your PATH.

## Metric Sources

The metric names and directions follow the pyIQA usage in the neighboring
`ThermalQualityMeasurement` project:

- FR examples: `gmsd`, `dists`, `lpips`, `ms_ssim`, `fsim`, `ssim`, `vif`, `psnr`,
  `topiq_fr`, `ahiq`, `ckdn`, `pieapp`
- NR examples: `brisque`, `niqe`, `ilniqe`, `piqe`, `nrqm`, `nima`, `cnniqa`

The app initializes metrics with:

```python
pyiqa.create_metric(metric_name, device=device)
```

and calls FR metrics as:

```python
score = metric(distorted_tensor, reference_tensor)
```
