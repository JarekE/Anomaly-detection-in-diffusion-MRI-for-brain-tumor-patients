# This will be my first basic model

import torch
from Models.BaseVAE import BaseVAE
from torch import nn
from torch.nn import functional as F
from config import Tensor
from typing import List
from config import ac_function_rec

"""

All Variational AutoEncoder:

- VanillaVAE (Basic, with flatted Latent Space in "normal" size)
- SpatialVAE (Spatial Latent Space with 512 Channels and a shape of 4x5x4)
- VoxelVAE (Single voxel pass the VAE. Afterwards the image will be reconstructed)

"""


class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # 40960 -> 512 -> 256
        self.fc = nn.Linear(hidden_dims[-1] * 4 * 5 * 4, hidden_dims[-1])
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4*5*4)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm3d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        if ac_function_rec == "Sigmoid":
            self.final_layer = nn.Sequential(
                            nn.ConvTranspose3d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm3d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv3d(hidden_dims[-1], out_channels= 64,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())
        else:
            self.final_layer = nn.Sequential(
                nn.ConvTranspose3d(hidden_dims[-1],
                                   hidden_dims[-1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm3d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv3d(hidden_dims[-1], out_channels=64,
                          kernel_size=3, padding=1))



    def encode(self, input: Tensor) -> List[Tensor]:

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.fc(result)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:

        result = self.decoder_input(z)
        result = result.view(-1, 512, 4, 5, 4)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:

        recons = args[0]
        #input = args[1]
        mu = args[2]
        log_var = args[3]
        target = args[4]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, target)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    # Not in use atm. Maybe use later.
    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the correspondings
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


"""
SpatialVAE: Latent space is 3D with an extremly larger shape
"""

class SpatialVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(SpatialVAE, self).__init__()

        #self.latent_dim = 2

        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Conv3d(hidden_dims[-1], out_channels=hidden_dims[-1],
                              kernel_size= 1, stride= 1, padding  = 0)
        self.fc_var = nn.Conv3d(hidden_dims[-1], out_channels=hidden_dims[-1],
                               kernel_size=1, stride=1, padding=0)

        #self.decoder_input = nn.ConvTranspose3d(hidden_dims[-1], hidden_dims[-1], kernel_size=1)

        # Build Decoder
        modules = []

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm3d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose3d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm3d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv3d(hidden_dims[-1], out_channels= 64,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())


    def encode(self, input: Tensor) -> List[Tensor]:

        result = self.encoder(input)

        # 512, 4, 6, 4
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:

        #result = self.decoder_input(z)
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:

        recons = args[0]
        #input = args[1]
        mu = args[2]
        log_var = args[3]
        target = args[4]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, target)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

"""
VoxelVAE: Fully connected layer with channels as 1-Dimension per voxel.
"""


class VoxelVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 **kwargs) -> None:
        super(VoxelVAE, self).__init__()

        out_features = in_channels
        modules = []
        if latent_dim is 2:
            hidden_dims = [32, 16, 8, 4]
        elif latent_dim is 4:
            hidden_dims = [32, 16, 8]
        elif latent_dim is 8:
            hidden_dims = [32, 16]
        else:
            hidden_dims = [32]

        # encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])
        hidden_dims.reverse()

        # decoder
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i + 1]),
                    nn.ReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(in_features=hidden_dims[-1], out_features=out_features))


    def encode(self, input: Tensor) -> List[Tensor]:

        result = self.encoder(input)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:

        result = self.decoder_input(z)
        result = self.decoder(result)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        z_decode = self.decode(z)

        return  [self.final_layer(z_decode), input, mu, log_var, z]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:

        recons = args[0]
        #input = args[1]
        mu = args[2]
        log_var = args[3]
        target = args[5]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, target)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}


class CNNVoxelVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(CNNVoxelVAE, self).__init__()

        # encoder
        self.conv_block = nn.Sequential(nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm3d(128),
                                        nn.LeakyReLU())

        self.encoder = nn.Sequential(nn.Linear(in_features=128*3*3*3, out_features=1000),
                                     nn.ReLU(),
                                     nn.Linear(in_features=1000, out_features=64),
                                     nn.ReLU(),
                                     nn.Linear(in_features=64, out_features=16),
                                     nn.ReLU(),
                                     nn.Linear(in_features=16, out_features=8),
                                     nn.ReLU(),
                                     )
        # decoder
        self.decoder = nn.Sequential(nn.Linear(in_features=8, out_features=16),
                                     nn.ReLU(),
                                     nn.Linear(in_features=16, out_features=64),
                                     nn.ReLU(),
                                     nn.Linear(in_features=64, out_features=1000),
                                     nn.ReLU(),
                                     nn.Linear(in_features=1000, out_features=64*1*1*1),
                                     nn.Sigmoid()
                                     )

        self.fc_mu = nn.Linear(in_features=8, out_features=2)
        self.fc_var = nn.Linear(in_features=8, out_features=2)

        self.decoder_input = nn.Linear(in_features=2, out_features=8)


    def encode(self, input: Tensor) -> List[Tensor]:

        conv = self.conv_block(input)
        flat = torch.flatten(conv, start_dim=1)
        result = self.encoder(flat)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:

        result = self.decoder_input(z)
        result = self.decoder(result)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:

        recons = args[0]
        #input = args[1]
        mu = args[2]
        log_var = args[3]
        target = args[4]

        # Only middle Voxel
        target = target[:,:,1,1,1]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, target)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
