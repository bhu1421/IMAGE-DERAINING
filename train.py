# Hybrid Image Deraining Training Script
# Phase 1: Train Autoencoder
# Phase 2: Train GAN (Pix2Pix)

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.rain100l_loader import Rain100LDataset
from models.unet_autoencoder import UNetAutoencoder
from models.pix2pix_gan import Pix2PixGenerator, Pix2PixDiscriminator
import numpy as np
from tqdm import tqdm


def get_dataloaders(batch_size=8, num_workers=0, pin_memory=False):
    transform = transforms.ToTensor()
    train_rainy = os.path.join('Rain100L', 'rain_data_train_Light', 'rain')
    train_clean = os.path.join('Rain100L', 'rain_data_train_Light', 'norain')
    test_rainy = os.path.join('Rain100L', 'rain_data_test_Light', 'rain', 'X2')
    test_clean = os.path.join('Rain100L', 'rain_data_test_Light', 'norain')
    train_set = Rain100LDataset(train_rainy, train_clean, transform=transform)
    test_set = Rain100LDataset(test_rainy, test_clean, transform=transform)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def load_checkpoint_if_available(model, checkpoint_path, label):
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"Resumed {label} from: {checkpoint_path}")
    return model


def train_autoencoder(train_loader, device, epochs=10, use_amp=False, checkpoint_path=None):
    model = UNetAutoencoder().to(device)
    model = load_checkpoint_if_available(model, checkpoint_path, "autoencoder")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"[Autoencoder] Epoch {epoch+1}/{epochs}")
        for rainy, clean in loop:
            rainy = rainy.to(device, non_blocking=use_amp)
            clean = clean.to(device, non_blocking=use_amp)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(rainy)
                loss = criterion(output, clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=loss.item())
    torch.save(model.state_dict(), 'outputs/autoencoder.pth')
    return model


def train_gan(
    train_loader,
    device,
    epochs=10,
    use_amp=False,
    generator_checkpoint=None,
    discriminator_checkpoint=None,
):
    G = Pix2PixGenerator().to(device)
    D = Pix2PixDiscriminator().to(device)
    G = load_checkpoint_if_available(G, generator_checkpoint, "generator")
    D = load_checkpoint_if_available(D, discriminator_checkpoint, "discriminator")
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    optimizer_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    scaler_G = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler_D = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"[GAN] Epoch {epoch+1}/{epochs}")
        for rainy, clean in loop:
            rainy = rainy.to(device, non_blocking=use_amp)
            clean = clean.to(device, non_blocking=use_amp)
            # Train Discriminator
            optimizer_D.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake = G(rainy)
                real_pair = D(rainy, clean)
                fake_pair = D(rainy, fake.detach())
                valid = torch.ones_like(real_pair)
                fake_label = torch.zeros_like(fake_pair)
                loss_D = (criterion_gan(real_pair, valid) + criterion_gan(fake_pair, fake_label)) * 0.5
            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()
            # Train Generator
            optimizer_G.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake = G(rainy)
                fake_pair = D(rainy, fake)
                valid = torch.ones_like(fake_pair)
                loss_G_gan = criterion_gan(fake_pair, valid)
                loss_G_l1 = criterion_l1(fake, clean)
                loss_G = loss_G_gan + 100 * loss_G_l1
            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()
            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())
    torch.save(G.state_dict(), 'outputs/pix2pix_generator.pth')
    torch.save(D.state_dict(), 'outputs/pix2pix_discriminator.pth')
    return G, D


def parse_args():
    parser = argparse.ArgumentParser(description="Train the hybrid image deraining models.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--epochs-autoencoder", type=int, default=10, help="Epochs for autoencoder training.")
    parser.add_argument("--epochs-gan", type=int, default=10, help="Epochs for GAN training.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument(
        "--skip-autoencoder",
        action="store_true",
        help="Skip autoencoder training. Useful when you only want to continue GAN training.",
    )
    parser.add_argument(
        "--skip-gan",
        action="store_true",
        help="Skip GAN training.",
    )
    parser.add_argument(
        "--resume-autoencoder",
        default=None,
        help="Path to autoencoder weights to continue training from.",
    )
    parser.add_argument(
        "--resume-generator",
        default=None,
        help="Path to generator weights to continue GAN training from.",
    )
    parser.add_argument(
        "--resume-discriminator",
        default=None,
        help="Path to discriminator weights to continue GAN training from.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA GPU not detected. Training will run on CPU.")

    os.makedirs('outputs', exist_ok=True)
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    if not args.skip_autoencoder:
        print("Training Autoencoder...")
        autoencoder = train_autoencoder(
            train_loader,
            device,
            epochs=args.epochs_autoencoder,
            use_amp=use_cuda,
            checkpoint_path=args.resume_autoencoder,
        )
    else:
        print("Skipping autoencoder training.")

    if not args.skip_gan:
        print("Training GAN...")
        gan_G, gan_D = train_gan(
            train_loader,
            device,
            epochs=args.epochs_gan,
            use_amp=use_cuda,
            generator_checkpoint=args.resume_generator,
            discriminator_checkpoint=args.resume_discriminator,
        )
    else:
        print("Skipping GAN training.")

    print("Training complete. Models saved in outputs/.")


if __name__ == "__main__":
    main()
