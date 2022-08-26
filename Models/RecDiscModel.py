import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import random
import matplotlib.pyplot as plt


class RecDisc(nn.Module):
    def __init__(self, in_channels: int, in_channels_unet: int):
        super().__init__()


        # Reconstruction Network
        latent_dim = 16

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
        self.fc_latenspace = nn.Linear(hidden_dims[-1], latent_dim)

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

        self.in_dim = in_channels_unet
        self.out_dim = 2
        self.num_filter = 16
        act_fn = nn.ReLU(inplace=True)

        #%% UNet body
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

    def encode(self, input):

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.fc(result)
        latentspace = self.fc_latenspace(result)
        return latentspace

    def decode(self, z):

        result = self.decoder_input(z)
        result = result.view(-1, 512, 4, 5, 4)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

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

    def anomaly_generation(self, input):
        # Generate anomaly block
        block = torch.zeros_like(input)
        x, y, z = 32, 40, 32
        random_x = random.randint(-9, 9) + x
        random_y = random.randint(-12, 12) + y
        random_z = random.randint(-9, 9) + z
        random_size = random.randint(4, 10)
        anomaly_block = torch.ones(random_size*2, random_size*2, random_size*2)

        # Random value for block
        anomaly_block = torch.mul(anomaly_block, random.uniform(-1, 1))

        # Place block
        block[:, :, random_x - random_size:random_x + random_size, random_y - random_size:random_y + random_size,
        random_z - random_size:random_z + random_size] = anomaly_block

        # Add block to input
        input = torch.add(input, block)

        # Reconstruction map
        reconstructive_map_tumor = torch.where(block[:,0,:,:,:] != 0, 1, 0)
        reconstructive_map_background = torch.where(block[:,0,:,:,:] == 0, 1, 0)
        reconstructive_map = torch.cat((reconstructive_map_tumor[:,None,:,:,:], reconstructive_map_background[:,None,:,:,:]), dim=1)

        """
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax[0].imshow(input.cpu().numpy()[2, 30, :, :, random_z], cmap='gray')
        ax[0].axis('off')
        ax[1].imshow(reconstructive_map.cpu().numpy()[2, 0, :, :, random_z], cmap='gray')
        ax[1].axis('off')
        plt.tight_layout()
        plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        ax[0].imshow(input.cpu().numpy()[1, 49, :, :, random_z], cmap='gray')
        ax[0].axis('off')
        ax[1].imshow(reconstructive_map.cpu().numpy()[1, 0, :, :, random_z], cmap='gray')
        ax[1].axis('off')
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        """
        return input, reconstructive_map, random_z

    def reconstruction(self, x):
        latentspace = self.encode(x)
        x = self.decode(latentspace)
        return x

    def recon_loss(self, reconstruction, input):
        loss = F.mse_loss(input, reconstruction)
        return loss

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

    def forward(self, x):

        anomaly_x, reconstructive_map, z = self.anomaly_generation(x)

        rec = self.reconstruction(anomaly_x)

        # Calculate reconstruction loss
        reconstruction_loss = self.recon_loss(rec, x)
        # Concatenate input and reconstructed image
        conc = torch.cat((rec, anomaly_x), dim=1)

        disc = self.discriminative(conc)
        return [disc, reconstruction_loss, reconstructive_map, z, x, anomaly_x, rec]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:

        discriminative_image = args[0]
        reconstruction_loss = args[1]
        reconstructive_map = args[2]
        #z = args[3]
        #input = args[4]

        # Segmentation loss
        discriminative_loss = F.l1_loss(discriminative_image, reconstructive_map)

        loss = reconstruction_loss + discriminative_loss

        return {'loss': loss, 'Reconstruction_Loss': reconstruction_loss.detach(), 'Discriminative_Loss': discriminative_loss.detach()}
