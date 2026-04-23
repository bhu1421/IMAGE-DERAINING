import os
from pathlib import Path

import cv2
import numpy as np
import torch

from src.deraining.config import IMAGE_SIZE


def load_image_for_inference(image_path):
    image_bgr = cv2.imread(os.fspath(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read input image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_size = (image_rgb.shape[1], image_rgb.shape[0])
    resized = cv2.resize(image_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return image_rgb, tensor, original_size


def tensor_to_uint8_image(tensor):
    image = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255.0).round().astype(np.uint8)


def tensor_to_rgb_image(tensor, output_size=None):
    image = tensor.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    if output_size is not None:
        image = cv2.resize(image, output_size, interpolation=cv2.INTER_CUBIC)
    return image


def save_rgb_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(os.fspath(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

