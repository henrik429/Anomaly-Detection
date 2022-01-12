import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    An encoder that down-sample the input x to latent representation z.
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.leakyReLU = nn.LeakyReLU()
        self.fc1 = nn.Linear(121, 64)
        self.fc2 = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = self.fc2(self.leakyReLU(self.fc1(x)))
        return x


class Decoder(nn.Module):
    """
    A Decoder representing the Generator that up-sample the latent representation z to sample x.
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 121)

    def forward(self, z):
        x = self.fc3(self.ReLU(self.fc2(self.ReLU(self.fc1(z)))))
        return x


class DiscriminatorXZ(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(p=0.5)

        self.leakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc_x = nn.Linear(121, 128)
        self.bn = nn.BatchNorm1d(128)
        self.fc_z = nn.Linear(latent_dim, 128)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, z):
        x = self.leakyReLU(self.bn(self.fc_x(x)))
        z = self.dropout(self.leakyReLU(self.fc_z(z)))

        xz = torch.cat((x, z), dim=1)
        xz = self.sigmoid(self.fc2(self.dropout(self.leakyReLU(self.fc1(xz)))))
        return xz


class DiscriminatorXX(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.leakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(242, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, rec_x):
        out = torch.cat((x, rec_x), dim=1)
        intermediate_layer = self.dropout(self.leakyReLU(self.fc1(out)))
        out = self.dropout(self.sigmoid(self.fc2(intermediate_layer)))
        return out, intermediate_layer


class DiscriminatorZZ(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.leakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(latent_dim*2, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, z, rec_z):
        out = torch.cat((z, rec_z), dim=1)
        out = self.dropout(self.leakyReLU(self.fc1(out)))
        out = self.sigmoid(self.dropout(self.fc2(out)))
        return out


if __name__ == '__main__':
    """
    Just for Testing.
    """

    enc = Encoder()
    enc(torch.randn((10, 121)))

    dec = Decoder()
    dec(torch.randn((10, 32)))

    dis_xz = DiscriminatorXZ()
    dis_xz(torch.randn((10, 121)), torch.randn((10, 32)))

    dis_xx = DiscriminatorXX()
    dis_xx(torch.randn((10, 121)), torch.randn((10, 121)))

    dis_zz = DiscriminatorZZ()
    dis_zz(torch.randn((10, 32)), torch.randn((10, 32)))