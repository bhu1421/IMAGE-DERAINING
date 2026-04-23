import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.deraining.config import (
    DISCRIMINATOR_CHECKPOINT,
    GENERATOR_CHECKPOINT,
    OUTPUTS_DIR,
    TRAIN_CLEAN_DIR,
    TRAIN_RAIN_DIR,
)
from src.deraining.data import Rain100LDataset
from src.deraining.models import Pix2PixDiscriminator, Pix2PixGenerator


def get_train_loader(batch_size=8, num_workers=0, pin_memory=False):
    transform = transforms.ToTensor()
    train_set = Rain100LDataset(TRAIN_RAIN_DIR, TRAIN_CLEAN_DIR, transform=transform)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader


def load_checkpoint_if_available(model, checkpoint_path, label):
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"Resumed {label} from: {checkpoint_path}")
    return model


def train_gan(
    train_loader,
    device,
    epochs=10,
    use_amp=False,
    generator_checkpoint=None,
    discriminator_checkpoint=None,
):
    generator = Pix2PixGenerator().to(device)
    discriminator = Pix2PixDiscriminator().to(device)
    generator = load_checkpoint_if_available(generator, generator_checkpoint, "generator")
    discriminator = load_checkpoint_if_available(discriminator, discriminator_checkpoint, "discriminator")

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    optimizer_g = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    scaler_g = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler_d = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"[GAN] Epoch {epoch + 1}/{epochs}")
        for rainy, clean in loop:
            rainy = rainy.to(device, non_blocking=use_amp)
            clean = clean.to(device, non_blocking=use_amp)

            optimizer_d.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake = generator(rainy)
                real_pair = discriminator(rainy, clean)
                fake_pair = discriminator(rainy, fake.detach())
                valid = torch.ones_like(real_pair)
                fake_label = torch.zeros_like(fake_pair)
                loss_d = (criterion_gan(real_pair, valid) + criterion_gan(fake_pair, fake_label)) * 0.5
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            optimizer_g.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake = generator(rainy)
                fake_pair = discriminator(rainy, fake)
                valid = torch.ones_like(fake_pair)
                loss_g_gan = criterion_gan(fake_pair, valid)
                loss_g_l1 = criterion_l1(fake, clean)
                loss_g = loss_g_gan + 100 * loss_g_l1
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()
            loop.set_postfix(loss_D=loss_d.item(), loss_G=loss_g.item())

    torch.save(generator.state_dict(), os.fspath(GENERATOR_CHECKPOINT))
    torch.save(discriminator.state_dict(), os.fspath(DISCRIMINATOR_CHECKPOINT))
    return generator, discriminator


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Pix2Pix deraining model.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for GAN training.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument(
        "--resume-generator",
        default=None,
        help="Path to generator weights to continue GAN training from.",
    )
    parser.add_argument(
        "--resume-discriminator",
        default=None,
        help="Path to discriminator weights to continue training from.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA GPU not detected. Training will run on CPU.")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    train_loader = get_train_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    print("Training Pix2Pix GAN...")
    train_gan(
        train_loader,
        device,
        epochs=args.epochs,
        use_amp=use_cuda,
        generator_checkpoint=args.resume_generator,
        discriminator_checkpoint=args.resume_discriminator,
    )

    print("Training complete. Generator and discriminator saved in outputs/.")


if __name__ == "__main__":
    main()
