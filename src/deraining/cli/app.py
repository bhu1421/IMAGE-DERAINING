import argparse
import os

import numpy as np
import torch

from src.deraining.config import APP_COMPARISON_IMAGE, APP_RESULT_IMAGE, GENERATOR_CHECKPOINT
from src.deraining.models import Pix2PixGenerator
from src.deraining.utils import load_image_for_inference, save_rgb_image, tensor_to_rgb_image


def parse_args():
    parser = argparse.ArgumentParser(description="Derain a single image using the trained Pix2Pix generator.")
    parser.add_argument("input", help="Path to the rainy input image.")
    parser.add_argument(
        "--weights",
        default=os.fspath(GENERATOR_CHECKPOINT),
        help="Path to the trained Pix2Pix generator weights.",
    )
    parser.add_argument(
        "--output",
        default=os.fspath(APP_RESULT_IMAGE),
        help="Path to save the derained output image.",
    )
    parser.add_argument(
        "--comparison",
        default=os.fspath(APP_COMPARISON_IMAGE),
        help="Path to save the rainy/derained comparison image.",
    )
    return parser.parse_args()


def build_model(weights_path, device):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Generator weights not found: {weights_path}")
    model = Pix2PixGenerator().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA GPU not detected. Inference will run on CPU.")

    rainy_rgb, input_tensor, original_size = load_image_for_inference(args.input)
    model = build_model(args.weights, device)

    with torch.inference_mode():
        derained_tensor = model(input_tensor.to(device, non_blocking=device.type == "cuda"))

    derained_rgb = tensor_to_rgb_image(derained_tensor, original_size)
    comparison_rgb = np.concatenate([rainy_rgb, derained_rgb], axis=1)

    save_rgb_image(args.output, derained_rgb)
    save_rgb_image(args.comparison, comparison_rgb)

    print(f"Saved derained image to: {args.output}")
    print(f"Saved comparison image to: {args.comparison}")


if __name__ == "__main__":
    main()
