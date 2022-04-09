"""
This file reproduces the architecture of the ALAD paper for input of image data.
"""

import torch
import torch.nn as nn
import numpy as np


def conv_block(
    in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True
):
    batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
    return [
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=padding,
            stride=(stride, stride),
        ),
        batch_norm,
        nn.LeakyReLU(),
    ]


def deconv_block(
    in_channels,
    out_channels,
    kernel_size=4,
    stride=2,
    padding=1,
    activation="ReLU",
    batch_norm=True,
):
    activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()
    batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
    return [
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding),
            stride=(stride, stride),
        ),
        batch_norm,
        activation,
    ]


class Encoder(nn.Module):
    """
    An encoder that down-sample the input x to latent representation z.
    """

    def __init__(self, latent_dim=100):
        super().__init__()

        self.conv1 = nn.Sequential(
            *conv_block(in_channels=3, out_channels=128, kernel_size=4, stride=2)
        )
        self.conv2 = nn.Sequential(
            *conv_block(in_channels=128, out_channels=256, kernel_size=4, stride=2)
        )
        self.conv3 = nn.Sequential(
            *conv_block(in_channels=256, out_channels=512, kernel_size=4, stride=2)
        )
        self.conv4 = nn.Conv2d(
            in_channels=512,
            out_channels=latent_dim,
            kernel_size=(4, 4),
            padding=0,
            stride=(1, 1),
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        z = self.conv4(x)
        z = z.view(-1, self.latent_dim)
        return z


class Decoder(nn.Module):
    """
    A Decoder representing the Generator that up-sample the latent representation z to sample x.
    """

    def __init__(self, latent_dim=100):
        super().__init__()

        self.conv1 = nn.Sequential(
            *deconv_block(
                in_channels=latent_dim,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=0,
            )
        )
        self.conv2 = nn.Sequential(
            *deconv_block(in_channels=512, out_channels=256, kernel_size=4, stride=2)
        )
        self.conv3 = nn.Sequential(
            *deconv_block(in_channels=256, out_channels=128, kernel_size=4, stride=2)
        )
        self.conv4 = nn.Sequential(
            *deconv_block(
                in_channels=128,
                out_channels=3,
                kernel_size=4,
                padding=1,
                stride=2,
                activation="Tanh",
            )
        )
        self.latent_dim = latent_dim

    def forward(self, z):
        z = z.reshape(z.shape[0], self.latent_dim, 1, 1)
        x = self.conv1(z)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class DiscriminatorXZ(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(p=0.2)

        self.conv1_x = nn.Sequential(
            *conv_block(
                in_channels=3,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                batch_norm=False,
            )
        )
        self.conv2_x = nn.Sequential(
            *conv_block(
                in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1
            )
        )
        self.conv3_x = nn.Sequential(
            *conv_block(
                in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1
            )
        )
        # Cut out batch normalization from conv blocks for z
        self.conv1_z = nn.Sequential(
            *conv_block(
                in_channels=latent_dim,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                batch_norm=False,
            )
        )
        self.conv2_z = nn.Sequential(
            *conv_block(
                in_channels=512,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                batch_norm=False,
            )
        )
        self.conv1_xz = nn.Sequential(
            *conv_block(
                in_channels=512 * 4 * 4 + 512,
                out_channels=1024,
                kernel_size=1,
                stride=1,
                padding=0,
                batch_norm=False,
            )
        )
        self.conv2_xz = nn.Conv2d(
            in_channels=1024,
            out_channels=1,
            kernel_size=(1, 1),
            padding=0,
            stride=(1, 1),
        )

    def forward(self, x, z):
        x = self.conv1_x(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)

        z = z.reshape(z.shape[0], self.latent_dim, 1, 1)

        z = self.dropout(self.conv1_z(z))
        z = self.dropout(self.conv2_z(z))

        # Concatenate x and z
        x = x.reshape(x.shape[0], np.prod(x.shape[1:]), 1, 1)
        xz = torch.cat((x, z), dim=1)

        xz = self.dropout(self.conv1_xz(xz))
        xz = self.conv2_xz(xz)

        xz = torch.squeeze(xz, dim=-1)
        xz = torch.squeeze(xz, dim=-1)

        return xz


class DiscriminatorXX(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(p=0.2)

        # Note: in the original paper padding would be "SAME" (i.e. padding is equal to 2).
        self.conv1 = nn.Sequential(
            *conv_block(6, 64, kernel_size=5, stride=2, padding=2, batch_norm=False)
        )
        self.conv2 = nn.Sequential(
            *conv_block(64, 128, kernel_size=5, stride=2, padding=2, batch_norm=False)
        )

        self.fc = nn.Linear(128 * 8 * 8, 1)

    def forward(self, x, x_):
        xx = torch.cat((x, x_), dim=1)
        xx = self.dropout(self.conv1(xx))
        xx = self.dropout(self.conv2(xx))

        intermediate_layer = xx.reshape(xx.shape[0], np.prod(xx.shape[1:]))

        xx = self.fc(intermediate_layer)
        return xx, intermediate_layer


class DiscriminatorZZ(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.leakyReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(2 * latent_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, z, z_):
        zz = torch.cat((z, z_), dim=1)
        zz = self.dropout(self.leakyReLU(self.fc1(zz)))
        zz = self.dropout(self.leakyReLU(self.fc2(zz)))
        zz = self.fc3(zz)
        return zz
