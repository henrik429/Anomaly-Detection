import torch
import torch.nn as nn
import numpy as np


def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size),
                      padding=padding, stride=(stride, stride)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels)]


def deconv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, activation='ReLU'):
    activation = nn.ReLU() if activation == 'ReLU' else nn.Tanh()
    return [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=(kernel_size, kernel_size),
                                  padding=(padding, padding), stride=(stride, stride)),
            activation,
            nn.BatchNorm2d(out_channels)]


class Encoder(nn.Module):
    """
    An encoder that
    """
    def __init__(self, latent_dim=100):
        super().__init__()

        self.conv1 = nn.Sequential(*conv_block(in_channels=3, out_channels=128, kernel_size=4, stride=2))
        self.conv2 = nn.Sequential(*conv_block(in_channels=128, out_channels=256, kernel_size=4, stride=2))
        self.conv3 = nn.Sequential(*conv_block(in_channels=256, out_channels=512, kernel_size=4, stride=2))
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=latent_dim, kernel_size=(4, 4),
                      padding=0, stride=(1, 1))
        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        z = self.conv4(x)
        z = z.view(-1, self.latent_dim)
        print(z.shape)
        return z


class Decoder(nn.Module):
    """
    A decoder which is basically the Generator.
    """
    def __init__(self, latent_dim=100):
        super().__init__()

        self.conv1 = nn.Sequential(*deconv_block(in_channels=latent_dim, out_channels=512, kernel_size=4, stride=2, padding=0))
        self.conv2 = nn.Sequential(*deconv_block(in_channels=512, out_channels=256, kernel_size=4, stride=2))
        self.conv3 = nn.Sequential(*deconv_block(in_channels=256, out_channels=128, kernel_size=4, stride=2))
        self.conv4 = nn.Sequential(*deconv_block(in_channels=128, out_channels=3, kernel_size=4, padding=1, stride=2,
                                                 activation="Tanh"))
        self.latent_dim = latent_dim

    def forward(self, z):
        z = z.reshape(z.shape[0], self.latent_dim, 1, 1)
        x = self.conv1(z)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        print(x.shape)
        return x


class DiscriminatorXZ(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(p=0.2)

        # Cut out batch normalization from conv block 1
        self.conv1_x = nn.Sequential(*conv_block(in_channels=3, out_channels=128, kernel_size=4, stride=2)[:2])

        self.conv2_x = nn.Sequential(*conv_block(in_channels=128, out_channels=256, kernel_size=4, stride=2))
        self.conv3_x = nn.Sequential(*conv_block(in_channels=256, out_channels=512, kernel_size=4, stride=2))

        # Cut out batch normalization from conv blocks for z
        self.conv1_z = nn.Sequential(*conv_block(in_channels=latent_dim, out_channels=512, kernel_size=1, stride=1,
                                                 padding=0)[:2])
        self.conv2_z = nn.Sequential(*conv_block(in_channels=512, out_channels=512, kernel_size=1, stride=1,
                                                 padding=0)[:2])

        self.conv1_xz = nn.Sequential(*conv_block(in_channels=8704, out_channels=1024, kernel_size=1, stride=1,
                                                  padding=0)[:2])
        self.conv2_xz = nn.Sequential(*conv_block(in_channels=1024, out_channels=1, kernel_size=1, stride=1,
                                                  padding=0)[:2])

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
        xz = self.dropout(self.conv2_xz(xz))

        xz = torch.squeeze(xz,dim=-1)
        xz = torch.squeeze(xz,dim=-1)
        print(xz.shape)

        return xz


class DiscriminatorXX(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(p=0.2)

        self.conv1 = nn.Sequential(*conv_block(6, 64, kernel_size=5, stride=2, padding=2)[:2])
        self.conv2 = nn.Sequential(*conv_block(64, 128, kernel_size=5, stride=2, padding=2)[:2])

        self.fc = nn.Linear(128*8*8, 1)

    def forward(self, x, x_):
        xx = torch.cat((x,x_), dim=1)
        xx = self.dropout(self.conv1(xx))
        xx = self.dropout(self.conv2(xx))
        xx = xx.reshape(xx.shape[0], np.prod(xx.shape[1:]))
        xx = self.fc(xx)
        #xx = xx.squeeze()
        print(xx.shape)
        return xx


class DiscriminatorZZ(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.leakyReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, z, z_):
        zz = torch.cat((z,z_), dim=1)
        zz = self.leakyReLU(self.dropout(self.fc1(zz)))
        zz = self.leakyReLU(self.dropout(self.fc2(zz)))
        zz = self.fc3(zz)

        print(zz.shape)
        return zz


if __name__ == '__main__':
    enc = Encoder()
    enc(torch.randn((10, 3, 32, 32)))

    dec = Decoder()
    dec(torch.randn((10, 100)))

    dis_xz = DiscriminatorXZ()
    dis_xz(torch.randn((10, 3, 32, 32)), torch.randn((10, 100)))

    dis_xx = DiscriminatorXX()
    dis_xx(torch.randn((10, 3, 32, 32)), torch.randn((10, 3, 32, 32)))

    dis_zz = DiscriminatorZZ()
    dis_zz(torch.randn((10, 100)), torch.randn((10, 100)))