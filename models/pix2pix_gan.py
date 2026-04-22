import torch
import torch.nn as nn

# Generator (U-Net style)
class Pix2PixGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Pix2PixGenerator, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.up1 = self.up_block(512, 256)
        self.up2 = self.up_block(256, 128)
        self.up3 = self.up_block(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d1 = self.up1(e4)
        d2 = self.up2(d1)
        d3 = self.up3(d2)
        out = self.final(d3)
        return torch.sigmoid(out)

# Discriminator (PatchGAN)
class Pix2PixDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super(Pix2PixDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, x, y):
        # x: input image, y: target or generated image
        inp = torch.cat([x, y], dim=1)
        return self.model(inp)
