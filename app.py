import argparse
import os

import cv2
import numpy as np
import torch

from models.pix2pix_gan import Pix2PixGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Derain a single image using the trained Pix2Pix generator.")
    parser.add_argument("input", help="Path to the rainy input image.")
    parser.add_argument(
        "--weights",
        default=os.path.join("outputs", "pix2pix_generator.pth"),
        help="Path to the trained Pix2Pix generator weights.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("outputs", "app_result.png"),
        help="Path to save the derained output image.",
    )
    parser.add_argument(
        "--comparison",
        default=os.path.join("outputs", "app_comparison.png"),
        help="Path to save the rainy/derained comparison image.",
    )
    return parser.parse_args()


def load_image(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read input image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_size = (image_rgb.shape[1], image_rgb.shape[0])
    resized = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return image_rgb, tensor, original_size


def tensor_to_rgb_image(tensor, output_size):
    image = tensor.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    if output_size is not None:
        image = cv2.resize(image, output_size, interpolation=cv2.INTER_CUBIC)
    return image


def save_rgb_image(path, image):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


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

    rainy_rgb, input_tensor, original_size = load_image(args.input)
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
