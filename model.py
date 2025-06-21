import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Hyperparameters
# -----------------------------
latent_dim = 20
batch_size = 128
num_epochs = 10
lr = 1e-3

# -----------------------------
# Dataset
# -----------------------------
transform = transforms.ToTensor()
mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# -----------------------------
# Model
# -----------------------------


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28 + 10, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

    def forward(self, x, y):
        x = x.view(-1, 28 * 28)
        y_onehot = torch.nn.functional.one_hot(y, num_classes=10).float()
        x = torch.cat([x, y_onehot], dim=1)
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + 10, 400)
        self.fc2 = nn.Linear(400, 28 * 28)

    def forward(self, z, y):
        y_onehot = torch.nn.functional.one_hot(y, num_classes=10).float()
        z = torch.cat([z, y_onehot], dim=1)
        h = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h)).view(-1, 1, 28, 28)


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, y), mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
