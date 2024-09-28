import torch
import torch.nn as nn


class outputbias(nn.Module):
    def __init__(self):
        super(outputbias, self).__init__()
        self.output_bias = nn.Parameter(torch.zeros(3, 32, 32), requires_grad=True)

    def forward(self, x):
        return x + self.output_bias


class TollAE(nn.Module):
    def __init__(self, encoder, decoder, dataset):
        super(TollAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dataset = dataset

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def compute_loss(self, sample, beta):
        recon, z = self.forward(sample)
        if self.dataset != 'arrhythmia':
            recon_loss = torch.mean(torch.norm(recon.view(recon.size(0), -1) -
                                               sample.view(sample.size(0), -1), dim=1))
        else:
            recon_loss = torch.mean(torch.norm(recon - sample, dim=1))
        z_norm = torch.norm(z, dim=1)
        loss = recon_loss + beta * z_norm.mean()
        return loss


def build_mnist_model(z_dim, dataset):

    encoder = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=4),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(64, 128, kernel_size=4),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(128, 256, kernel_size=4, stride=2),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(256, 512, kernel_size=4),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(512, 64, kernel_size=4),
        nn.Flatten(start_dim=1),
        nn.Linear(64 * 16, z_dim)
    )

    decoder = nn.Sequential(
        nn.Linear(z_dim, 64 * 16),
        nn.Unflatten(dim=1, unflattened_size=(64, 4, 4)),
        nn.ConvTranspose2d(64, 512, kernel_size=4),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(negative_slope=0.2),
        nn.ConvTranspose2d(512, 256, kernel_size=4),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=0.2),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2),
        nn.ConvTranspose2d(128, 64, kernel_size=4),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2),
        nn.ConvTranspose2d(64, 1, kernel_size=4)
    )

    return TollAE(encoder, decoder, dataset)


def build_cifar_model(z_dim, dataset):

    encoder = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(64, 128, kernel_size=4, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(256, 512, kernel_size=4, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(512, 512, kernel_size=1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(512, z_dim, kernel_size=1),
        nn.Flatten(start_dim=1)
    )

    decoder = nn.Sequential(
        nn.Unflatten(dim=1, unflattened_size=(z_dim, 1, 1)),
        nn.ConvTranspose2d(z_dim, 256, kernel_size=4, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=0.1),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.1),
        nn.ConvTranspose2d(128, 64, kernel_size=4, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.1),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(negative_slope=0.1),
        nn.ConvTranspose2d(32, 32, kernel_size=5, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(32, 32, kernel_size=1, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(32, 3, kernel_size=1, bias=False),
        outputbias()
    )

    return TollAE(encoder, decoder, dataset)


def build_arrhythmia_model(z_dim, dataset):
    
    encoder = nn.Sequential(
        nn.Linear(274, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, z_dim)
    )

    decoder = nn.Sequential(
        nn.Linear(z_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 274)
    )

    return TollAE(encoder, decoder, dataset)