import argparse
import os

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.deraining.config import GENERATOR_CHECKPOINT, TEST_CLEAN_DIR, TEST_RAIN_DIR, TEST_RESULTS_DIR
from src.deraining.data import Rain100LDataset
from src.deraining.models import Pix2PixGenerator
from src.deraining.utils import save_rgb_image, tensor_to_uint8_image


def parse_args():
    parser = argparse.ArgumentParser(description="Run deraining inference on the Rain100L test split.")
    parser.add_argument(
        "--weights",
        default=os.fspath(GENERATOR_CHECKPOINT),
        help="Path to the trained Pix2Pix generator weights.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.fspath(TEST_RESULTS_DIR),
        help="Directory where derained images and metrics will be saved.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    return parser.parse_args()


def get_test_loader(num_workers=0, pin_memory=False):
    transform = transforms.ToTensor()
    test_set = Rain100LDataset(TEST_RAIN_DIR, TEST_CLEAN_DIR, transform=transform)
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_set, test_loader


def get_rainy_image_name(dataset, idx):
    if hasattr(dataset, "image_pairs"):
        return dataset.image_pairs[idx][0].name
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_idx = dataset.indices[idx]
        return get_rainy_image_name(dataset.dataset, base_idx)
    raise AttributeError("Dataset does not expose image pair metadata.")


def evaluate_model(model, dataset, dataloader, device, output_dir):
    derained_dir = os.path.join(output_dir, "derained")
    comparison_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(derained_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    psnr_scores = []
    ssim_scores = []

    model.eval()
    with torch.inference_mode():
        for idx, (rainy, clean) in enumerate(tqdm(dataloader, desc="Testing")):
            rainy = rainy.to(device, non_blocking=device.type == "cuda")
            clean = clean.to(device, non_blocking=device.type == "cuda")
            derained = model(rainy)

            rainy_img = tensor_to_uint8_image(rainy[0])
            clean_img = tensor_to_uint8_image(clean[0])
            derained_img = tensor_to_uint8_image(derained[0])

            psnr = peak_signal_noise_ratio(clean_img, derained_img, data_range=255)
            ssim = structural_similarity(clean_img, derained_img, channel_axis=2, data_range=255)
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

            rainy_name = get_rainy_image_name(dataset, idx)
            output_name = rainy_name.replace("x2", "_derained")
            comparison_name = rainy_name.replace(".png", "_comparison.png")

            comparison = np.concatenate([rainy_img, derained_img, clean_img], axis=1)
            save_rgb_image(os.path.join(derained_dir, output_name), derained_img)
            save_rgb_image(os.path.join(comparison_dir, comparison_name), comparison)

    mean_psnr = float(np.mean(psnr_scores))
    mean_ssim = float(np.mean(ssim_scores))
    return mean_psnr, mean_ssim


def write_metrics(output_dir, mean_psnr, mean_ssim):
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        metrics_file.write(f"Average PSNR: {mean_psnr:.4f}\n")
        metrics_file.write(f"Average SSIM: {mean_ssim:.4f}\n")
    return metrics_path


def main():
    args = parse_args()
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Generator weights not found: {args.weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA GPU not detected. Testing will run on CPU.")

    os.makedirs(args.output_dir, exist_ok=True)
    dataset, test_loader = get_test_loader(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = Pix2PixGenerator().to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)

    mean_psnr, mean_ssim = evaluate_model(model, dataset, test_loader, device, args.output_dir)
    metrics_path = write_metrics(args.output_dir, mean_psnr, mean_ssim)

    print(f"Saved derained outputs to: {os.path.join(args.output_dir, 'derained')}")
    print(f"Saved comparisons to: {os.path.join(args.output_dir, 'comparisons')}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Average PSNR: {mean_psnr:.4f}")
    print(f"Average SSIM: {mean_ssim:.4f}")


if __name__ == "__main__":
    main()
