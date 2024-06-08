import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.nn import functional as F


def process_audio_config(config):
    """ Utility to read a *.ini file"""
    sampling_rate = config['audio'].getint('sampling_rate')
    hop_length = config['audio'].getint('hop_length')
    bins_per_octave = config['audio'].getint('bins_per_octave')
    num_octaves = config['audio'].getint('num_octaves')
    n_bins = int(num_octaves * bins_per_octave)
    n_iter = config['audio'].getint('n_iter')
    cqt_bit_depth = config['audio'].get('cqt_bit_depth')
    if cqt_bit_depth == "float64":
        torch.set_default_dtype(torch.float64)
        dtype = np.float64
    elif cqt_bit_depth == "float64":
        torch.set_default_dtype(torch.float32)
        dtype = np.float32
    else:
        raise TypeError('{} cqt_bit_depth datatype is unknown. Choose either float32 or float64'.format(cqt_bit_depth))
    return sampling_rate, hop_length, bins_per_octave, num_octaves, n_bins, n_iter, cqt_bit_depth, dtype


class VAE(nn.Module):
    def __init__(self, n_bins, n_units, latent_dim):
        super(VAE, self).__init__()

        self.n_bins = n_bins
        self.n_units = n_units
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(n_bins, n_units)
        self.fc2 = nn.Linear(n_units, 2 * latent_dim)
        self.fc3 = nn.Linear(latent_dim, n_units)
        self.fc4 = nn.Linear(n_units, n_bins)

    def encode(self, x):
        h = self.fc2(F.relu(self.fc1(x)))
        return h[:, 0:self.latent_dim], h[:, self.latent_dim:]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.relu(self.fc4(h3))

    def forward(self, x):
        x = x.view(-1, self.n_bins)  # Merge all CQT frames from all minibatch items into a single large "CQT-spectrogram"
        mu, logvar = self.encode(x)  # Batch dimension has disappeared at this point
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate_audio_from_CQT(self, x_cqt, **kwargs):
        """ Auto-encodes the given CQTs and also generates audio using Griffin-Lim phase reconstruction.
        :param kwargs: passed to librosa.griffinlim_cqt
        """
        # The non-flattened CQT shape is required to eventually reconstruct batch elements
        z_mu, _ = self.encode(x_cqt.view(-1, self.n_bins))
        return self.generate_audio_from_latent(z_mu, x_cqt.shape[0], **kwargs), z_mu

    def generate_audio_from_latent(self, z_flattened, minibatch_size: int, **kwargs):
        assert len(z_flattened.shape) == 2, "Expected a flattened z (batch dim merged with time dim)"
        decoded_cqt = self.decode(z_flattened).view(minibatch_size, -1, self.n_bins)
        y_inv_32 = [
            librosa.griffinlim_cqt(decoded_cqt[i, :, :].permute(1, 0).cpu().numpy(), **kwargs) for i in range(minibatch_size)
        ]
        return y_inv_32



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, kl_beta, n_bins):
    recon_loss = F.mse_loss(recon_x, x.view(-1, n_bins))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, (kl_beta * KLD)

