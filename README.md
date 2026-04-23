# IMAGE-DERAINING

Single-image deraining project built with PyTorch using the Rain100L dataset. The repository trains a Pix2Pix-style GAN with a U-Net generator, evaluates the trained generator on the test set, and runs single-image inference for demos.

## Project Overview

- Dataset: Rain100L
- Framework: PyTorch
- Models:
  - U-Net generator
  - PatchGAN discriminator
- Workflows:
  - training
  - resumed GAN training from saved checkpoints
  - test-set evaluation with PSNR and SSIM
  - single-image deraining demo

## Clean Repository Structure

```text
IMAGE-DERAINING/
|-- app.py
|-- train.py
|-- test.py
|-- requirements.txt
|-- README.md
|-- docs/
|   |-- PROJECT_STRUCTURE.md
|-- src/
|   |-- deraining/
|   |   |-- config.py
|   |   |-- cli/
|   |   |   |-- app.py
|   |   |   |-- test.py
|   |   |   |-- train.py
|   |   |-- data/
|   |   |   |-- rain100l.py
|   |   |-- models/
|   |   |   |-- pix2pix.py
|   |   |-- utils/
|   |   |   |-- images.py
|-- assets/
|   |-- results/
|-- outputs/                # generated locally, not tracked
|-- Rain100L/               # dataset locally, not tracked
```

## What Each Part Does

- `src/deraining/data/`: loads and pairs rainy and clean images
- `src/deraining/models/`: defines the U-Net generator and PatchGAN discriminator
- `src/deraining/utils/`: keeps shared image helper functions
- `src/deraining/cli/`: contains the actual training, testing, and inference workflows
- `train.py`, `test.py`, `app.py`: small entry scripts so commands stay simple
- `outputs/`: stores checkpoints, generated images, and metrics
- `docs/PROJECT_STRUCTURE.md`: short explanation you can use in presentation or viva

## Features

- Correct rainy/clean image pairing based on Rain100L image IDs
- GPU training and inference with CUDA
- Mixed precision training when CUDA is available
- Resume support for GAN training from saved checkpoints
- Test pipeline that saves derained images and comparison outputs
- Simple CLI app for deraining one custom image

## Setup

Create and use a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128
```

## Dataset

Place the Rain100L dataset in the project root:

```text
Rain100L/
|-- rain_data_train_Light/
|   |-- rain/
|   |-- norain/
|-- rain_data_test_Light/
|   |-- rain/X2/
|   |-- norain/
```

The dataset itself is not included in this repository.

## Training

Train the Pix2Pix GAN:

```powershell
.venv\Scripts\python.exe train.py --batch-size 8 --epochs 10 --num-workers 2
```

Continue GAN training from existing checkpoints:

```powershell
.venv\Scripts\python.exe train.py --resume-generator outputs\pix2pix_generator.pth --resume-discriminator outputs\pix2pix_discriminator.pth --epochs 20 --batch-size 8 --num-workers 2
```

Saved checkpoints:

- `outputs/pix2pix_generator.pth`
- `outputs/pix2pix_discriminator.pth`

## Testing

Run evaluation on the Rain100L test split:

```powershell
.venv\Scripts\python.exe test.py --num-workers 2
```

This generates:

- `outputs/test_results/derained/`
- `outputs/test_results/comparisons/`
- `outputs/test_results/metrics.txt`

Latest recorded metrics from the local run:

- `Average PSNR: 22.5720`
- `Average SSIM: 0.6804`

## Demo App

Run deraining on a single image:

```powershell
.venv\Scripts\python.exe app.py path\to\your\rainy_image.png
```

Example:

```powershell
.venv\Scripts\python.exe app.py Rain100L\rain_data_test_Light\rain\X2\norain-1x2.png
```

Optional output paths:

```powershell
.venv\Scripts\python.exe app.py Rain100L\rain_data_test_Light\rain\X2\norain-1x2.png --output outputs\my_result.png --comparison outputs\my_compare.png
```

## Sample Results

Single-image demo result:

![Single Image Demo Comparison](assets/results/sample_app_comparison.png)

Generated derained output:

![Single Image Demo Output](assets/results/sample_app_result.png)

Test-set comparison example:

![Test Set Comparison](assets/results/test_comparison_100.png)

## Notes

- The current inference pipeline uses the trained Pix2Pix generator.
- `outputs/`, `.venv/`, and `Rain100L/` are intentionally excluded from Git tracking.
