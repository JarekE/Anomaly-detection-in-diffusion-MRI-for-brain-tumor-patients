import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import config
import raster_geometry as rg
import warnings
from typing import Optional

from kornia.core import Tensor
from kornia.testing import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.utils.one_hot import one_hot


def anomaly_generation(input, mask):

    # Generate anomaly block
    input_anomaly = input
    sign = [-1, 1][random.randrange(2)]
    x, y, z = 32, 40, 32
    random_size = random.randint(5, 10)
    anomaly_block = torch.from_numpy(rg.sphere(2*random_size, random_size).astype(int))

    # Isotropic
    if random.uniform(0, 1) > 0.65:
        anomaly_block = torch.mul(anomaly_block, random.uniform(0, 1))
    # Random noise with centre value between 0.25 and 0.55
    else:
        gaussian_noise = np.random.normal(random.uniform(0.25, 0.55), 0.1, size=anomaly_block.shape)
        anomaly_block = torch.mul(anomaly_block, torch.from_numpy(gaussian_noise))

    # Place block in input-shaped array
    block = torch.zeros_like(input)
    random_x = sign * random.randint(4, 20) + x
    random_y = sign * random.randint(4, 20) + y
    random_z = sign * random.randint(4, 20) + z
    block[:, :, random_x - random_size:random_x + random_size, random_y - random_size:random_y + random_size,
    random_z - random_size:random_z + random_size] = anomaly_block

    # Check if point of approximate centre is inside the brainmask (should be improved)
    while not torch.equal(mask[:, random_x, random_y, random_z].cpu(), torch.ones(4)):
        random_x = sign * random.randint(4, 20) + x
        random_y = sign * random.randint(4, 20) + y
        random_z = sign * random.randint(4, 20) + z

        block = torch.zeros_like(input)
        block[:, :, random_x - random_size:random_x + random_size, random_y - random_size:random_y + random_size,
        random_z - random_size:random_z + random_size] = anomaly_block


    # Make space for the block (zero all elements)
    input_anomaly = torch.mul(input_anomaly, torch.where(block > 0, 0, 1))

    # Add block to input, not for testdata
    if config.mode == "test":
        input = input
    else:
        input = torch.add(input_anomaly, block)

    # Reconstruction map
    reconstructive_map = torch.where(block[:, 0, :, :, :] != 0, 1, 0)
    reconstructive_map = reconstructive_map[:, None, :, :, :]
    #reconstructive_map_background = torch.where(block[:, 0, :, :, :] == 0, 1, 0)
    #reconstructive_map = torch.cat((reconstructive_map_tumor[:, None, :, :, :], reconstructive_map_background[:, None, :, :, :]), dim=1)

    return input, reconstructive_map, random_z

def binary_focal_loss_with_logits(
    input: Tensor,
    target: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: Optional[float] = None,
    pos_weight: Optional[Tensor] = None,
) -> Tensor:

    KORNIA_CHECK_SHAPE(input, ["B", "C", "*"])
    KORNIA_CHECK(
        input.shape[0] == target.shape[0],
        f'Expected input batch_size ({input.shape[0]}) to match target batch_size ({target.shape[0]}).',
    )

    if pos_weight is None:
        pos_weight = torch.ones(input.shape[-1], device=input.device, dtype=input.dtype)

    KORNIA_CHECK_IS_TENSOR(pos_weight)
    KORNIA_CHECK(input.shape[-1] == pos_weight.shape[0], "Expected pos_weight equals number of classes.")

    probs_pos = input.sigmoid()
    probs_neg = torch.sigmoid(-input)

    loss_tmp = (
        -alpha * pos_weight * probs_neg.pow(gamma) * target * input.sigmoid().log()
        - (1 - alpha) * torch.pow(probs_pos, gamma) * (1.0 - target) * (-input).sigmoid().log()
    )

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss

class RecDisc(nn.Module):
    def __init__(self, in_channels: int, in_channels_unet: int):
        super().__init__()

        # Reconstruction Network
        latent_dim = 64

        modules = []
        hidden_dims = [64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
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

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4 * 5 * 4)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
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
            nn.Conv3d(hidden_dims[-1], out_channels=64,
                      kernel_size=3, padding=1),
            nn.Sigmoid())

        # UNet
        self.in_dim = in_channels_unet
        self.out_dim = 1
        self.num_filter = 16
        act_fn = nn.ReLU(inplace=True)

        self.down_1 = self.double_conv_block(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_2 = self.double_conv_block(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_3 = self.double_conv_block(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.bridge = self.double_conv_block(self.num_filter * 4, self.num_filter * 8, act_fn)

        self.trans_1 = self.up_conv(self.num_filter * 8, self.num_filter * 8, act_fn)
        self.up_1 = self.double_conv_block(self.num_filter * 12, self.num_filter * 4, act_fn)

        self.trans_2 = self.up_conv(self.num_filter * 4, self.num_filter * 4, act_fn)
        self.up_2 = self.double_conv_block(self.num_filter * 6, self.num_filter * 2, act_fn)

        self.trans_3 = self.up_conv(self.num_filter * 2, self.num_filter * 2, act_fn)
        self.up_3 = self.double_conv_block(self.num_filter * 3, self.num_filter, act_fn)

        self.out = self.out_block(self.num_filter, self.out_dim)

    def conv_block(self, in_dim, out_dim, act_fn):
        model = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_dim),
            act_fn,
        )
        return model

    def up_conv(self, in_dim, out_dim, act_fn):
        model = nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(out_dim),
            act_fn,
        )
        return model

    def double_conv_block(self, in_dim, out_dim, act_fn):
        model = nn.Sequential(
            self.conv_block(in_dim, out_dim, act_fn),
            self.conv_block(out_dim, out_dim, act_fn),
        )
        return model

    def out_block(self, in_dim, out_dim):
        model = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
            )
        return model

    def discriminative(self, x):
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        bridge = self.bridge(pool_3)

        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_3], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_2], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_1], dim=1)
        up_3 = self.up_3(concat_3)

        out = self.out(up_3)
        return out

    def encode(self, input):

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.fc(result)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):

        result = self.decoder_input(z)
        result = result.view(-1, 512, 4, 5, 4)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, mask, **kwargs):

        # Anomalys
        anomaly_x, reconstructive_map, z_value = anomaly_generation(input, mask)

        # Reconstruction
        mu, log_var = self.encode(anomaly_x)
        z = self.reparameterize(mu, log_var)
        rec = self.decode(z)

        # Concatenate input and reconstructed image
        conc = torch.cat((rec, anomaly_x), dim=1)

        disc = self.discriminative(conc)

        return [disc, input, mu, log_var, anomaly_x, reconstructive_map, z_value, rec]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:

        # Reconstruction loss
        recons = args[7]
        target = args[1]
        mu = args[2]
        log_var = args[3]
        reconstructive_map = args[5]
        discriminative_image = args[0]

        kld_weight = 0.0000122
        recons_loss = F.mse_loss(recons, target)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        reconstruction_loss = recons_loss + kld_weight * kld_loss

        # Segmentation loss
        #discriminative_loss = F.l1_loss(discriminative_image, reconstructive_map)
        kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        discriminative_loss = binary_focal_loss_with_logits(discriminative_image, reconstructive_map, **kwargs)


        loss = reconstruction_loss + discriminative_loss

        return {'loss': loss, 'Reconstruction_Loss': reconstruction_loss.detach(),
                'Discriminative_Loss': discriminative_loss.detach()}


"""

NEW CLASS
Uses Unets in both tasks

"""


class RecDiscUnet(nn.Module):
    def __init__(self, in_channels: int, in_channels_unet: int):
        super().__init__()

        self.num_filter = 16
        act_fn = nn.ReLU(inplace=True)

        # Reconstruction Network
        self.in_dim_rec = in_channels
        self.out_dim_rec = 64

        self.down_1_rec = self.double_conv_block(self.in_dim_rec, self.num_filter, act_fn)
        self.pool_1_rec = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_2_rec = self.double_conv_block(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2_rec = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_3_rec = self.double_conv_block(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3_rec = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.bridge_rec = self.double_conv_block(self.num_filter * 4, self.num_filter * 8, act_fn)

        self.trans_1_rec = self.up_conv(self.num_filter * 8, self.num_filter * 8, act_fn)
        self.up_1_rec = self.double_conv_block(self.num_filter * 12, self.num_filter * 4, act_fn)

        self.trans_2_rec = self.up_conv(self.num_filter * 4, self.num_filter * 4, act_fn)
        self.up_2_rec = self.double_conv_block(self.num_filter * 6, self.num_filter * 2, act_fn)

        self.trans_3_rec = self.up_conv(self.num_filter * 2, self.num_filter * 2, act_fn)
        self.up_3_rec = self.double_conv_block(self.num_filter * 3, self.num_filter, act_fn)

        self.out_rec = self.out_block(self.num_filter, self.out_dim_rec)

        # Discrimination Network
        self.in_dim = in_channels_unet
        self.out_dim = 1

        self.down_1 = self.double_conv_block(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_2 = self.double_conv_block(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_3 = self.double_conv_block(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.bridge = self.double_conv_block(self.num_filter * 4, self.num_filter * 8, act_fn)

        self.trans_1 = self.up_conv(self.num_filter * 8, self.num_filter * 8, act_fn)
        self.up_1 = self.double_conv_block(self.num_filter * 12, self.num_filter * 4, act_fn)

        self.trans_2 = self.up_conv(self.num_filter * 4, self.num_filter * 4, act_fn)
        self.up_2 = self.double_conv_block(self.num_filter * 6, self.num_filter * 2, act_fn)

        self.trans_3 = self.up_conv(self.num_filter * 2, self.num_filter * 2, act_fn)
        self.up_3 = self.double_conv_block(self.num_filter * 3, self.num_filter, act_fn)

        self.out = self.out_block(self.num_filter, self.out_dim)

    def conv_block(self, in_dim, out_dim, act_fn):
        model = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_dim),
            act_fn,
        )
        return model

    def up_conv(self, in_dim, out_dim, act_fn):
        model = nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(out_dim),
            act_fn,
        )
        return model

    def double_conv_block(self, in_dim, out_dim, act_fn):
        model = nn.Sequential(
            self.conv_block(in_dim, out_dim, act_fn),
            self.conv_block(out_dim, out_dim, act_fn),
        )
        return model

    def out_block(self, in_dim, out_dim):
        model = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
            )
        return model

    def reconstructive(self, x):
        down_1 = self.down_1_rec(x)
        pool_1 = self.pool_1_rec(down_1)
        down_2 = self.down_2_rec(pool_1)
        pool_2 = self.pool_2_rec(down_2)
        down_3 = self.down_3_rec(pool_2)
        pool_3 = self.pool_3_rec(down_3)

        bridge = self.bridge_rec(pool_3)

        trans_1 = self.trans_1_rec(bridge)
        concat_1 = torch.cat([trans_1, down_3], dim=1)
        up_1 = self.up_1_rec(concat_1)
        trans_2 = self.trans_2_rec(up_1)
        concat_2 = torch.cat([trans_2, down_2], dim=1)
        up_2 = self.up_2_rec(concat_2)
        trans_3 = self.trans_3_rec(up_2)
        concat_3 = torch.cat([trans_3, down_1], dim=1)
        up_3 = self.up_3_rec(concat_3)

        out = self.out_rec(up_3)
        return out

    def discriminative(self, x):
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        bridge = self.bridge(pool_3)

        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_3], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_2], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_1], dim=1)
        up_3 = self.up_3(concat_3)

        out = self.out(up_3)
        return out

    def forward(self, x, mask):

        anomaly_x, reconstructive_map, z = anomaly_generation(x, mask)

        rec = self.reconstructive(anomaly_x)

        # Concatenate input and reconstructed image
        conc = torch.cat((rec, anomaly_x), dim=1)

        disc = self.discriminative(conc)

        return [disc, reconstructive_map, z, x, anomaly_x, rec]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:

        discriminative_image = args[0]
        reconstructive_map = args[1]
        input = args[3]
        reconstruction = args[5]

        # Calculate reconstruction loss
        reconstruction_loss = F.mse_loss(input, reconstruction)

        # Segmentation loss
        discriminative_loss = F.l1_loss(discriminative_image, reconstructive_map)

        loss = reconstruction_loss + discriminative_loss

        return {'loss': loss, 'Reconstruction_Loss': reconstruction_loss.detach(), 'Discriminative_Loss': discriminative_loss.detach()}
