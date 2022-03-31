from pyparsing import Forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from zmq import device

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        out = input.view(self.shape)
        return out

def reparam(mu, log_var, size):
    epsilon = Variable(torch.randn(size)).to(device)
    return mu + torch.exp(log_var/2) * epsilon

class CNNBetaVAE(nn.Module):
    def __init__(self, latent_dim=32, in_channels=3):
        super(CNNBetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels,32,4,2),
            nn.ReLU(True),
            nn.Conv2d(32,32,4,2),
            nn.ReLU(True),
            nn.Conv2d(32,64,4,2),
            nn.ReLU(True),
            nn.Conv2d(64,64,4,2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(256, latent_dim*2)
        )
        # Reversed Encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            View((-1, 256, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,64,4,2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,64,4,2,1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,32,4,2,1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,in_channels,4,2,1)
        )

    def forward(self, x):
        outs = self.encoder(x)
        mu = outs[:,:self.latent_dim]
        log_var = outs[:,self.latent_dim:]
        size = log_var.shape
        z = reparam(mu, log_var, size)
        x_rec = self.decoder(z)
        return x_rec, mu, log_var


class FCBetaVAE(nn.Module):
    def __init__(self, latent_dim=10, input_dim=4096):
        super(FCBetaVAE).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1200),
            nn.ReLU(True),
            nn.Linear(1200, self.latent_dim*2),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 1200),
            nn.Tanh(True),
            nn.Linear(1200, 1200),
            nn.Tanh(True),
            nn.Linear(1200, 1200),
            nn.Tanh(True),
            nn.Linear(1200, 4096)
        )

    def forward(self, x):
        outs = self.encoder(x)
        mu = outs[:,:self.latent_dim]
        log_var = outs[:,self.latent_dim:]
        size = log_var.shape
        z = reparam(mu, log_var, size)
        x_rec = self.decoder(z)
        return x_rec, mu, log_var

