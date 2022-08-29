# Here I will define all my deep learning
import torch
from pytorch_lightning.core.lightning import LightningModule
from Models.VAE import VanillaVAE, SpatialVAE, VoxelVAE, CNNVoxelVAE
from Models.RecDiscModel import RecDisc, RecDiscUnet
from torch import optim
import numpy as np
from os.path import join as opj
from Models.UNet import UNet3d
import pickle
import config
from DataProcessing.data_visualization import show_RecDisc


class LearningModule(LightningModule):

    def __init__(self):
        super(LearningModule, self).__init__()

        # Choose from available networks
        if config.network == "VanillaVAE":
            self.model = VanillaVAE(in_channels=64, latent_dim=config.latent_dim)
            self.params = config.vanilla_params
        elif config.network == "SpatialVAE":
            self.model = SpatialVAE(in_channels=64)
            self.params = config.spatialvae_params
        elif config.network == "UNet":
            self.model = UNet3d(in_channels=64)
            self.params = config.unet_params
        elif config.network == "VoxelVAE":
            self.model = VoxelVAE(in_channels=64)
            self.params = config.voxelvae_params
        elif config.network == "CNNVoxelVAE":
            self.model = CNNVoxelVAE(in_channels=64)
            self.params = config.cnnvoxelvae_params
        elif config.network == "RecDisc":
            self.model = RecDisc(in_channels=64, in_channels_unet=64*2)
            self.params = config.rec_disc_params
        elif config.network == "RecDiscUnet":
            self.model = RecDiscUnet(in_channels=64, in_channels_unet=64*2)
            self.params = config.rec_disc_params
        else:
            raise ValueError('You chose a network that is not available atm: '+config.network)


    def forward(self, z, mask):
        y = self.model(z, mask)

        return y

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        target = batch['target']
        input = batch['input']
        mask = batch['mask_withoutCSF']

        if (config.network == "RecDisc") or (config.network == "RecDiscUnet"):
            results = self.forward(input, mask)
            train_loss = self.model.loss_function(*results)
        else:
            results = self.forward(input, mask)
            results.append(target)
            train_loss = self.model.loss_function(*results, M_N=self.params['kld_weight'])

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        target = batch['target']
        input = batch['input']
        mask = batch['mask_withoutCSF']

        if config.network == "RecDisc" or (config.network == "RecDiscUnet"):
            results = self.forward(input, mask)
            val_loss = self.model.loss_function(*results)
            if config.network == "RecDisc" and ((self.trainer.current_epoch < 30) or ((self.trainer.current_epoch/10).is_integer())):
                show_RecDisc(results[1], results[4], results[5], results[0], results[6], results[7])
            if config.network == "RecDiscUnet" and ((self.trainer.current_epoch < 30) or ((self.trainer.current_epoch/10).is_integer())):
                # [disc, reconstructive_map, z, x, anomaly_x, rec]
                show_RecDisc(results[3], results[4], results[1], results[0], results[2], results[5])
        else:
            results = self.forward(input, mask)
            results.append(target)
            val_loss = self.model.loss_function(*results, M_N=self.params['kld_weight'])

        print(val_loss.items())
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def test_step(self, batch, batch_idx):
        #labels = batch['target']
        real_img = batch['input']
        id_list = batch['id']
        mask = batch['mask_withoutCSF']

        # The mask is not in use here atm.
        results = self.forward(real_img, mask)

        if (config.network == "VoxelVAE") or (config.network == "CNNVoxelVAE"):
            coordinates_list = batch["coordinates"]
            # Result Values, ID of each value, Coordinates of each value, mu of each value, logvar of each value
            save_list = [results[0], id_list, coordinates_list, results[2], results[3]]

            with open('logs/DataDropOff/batch_'+str(batch_idx), 'wb') as fp:
                pickle.dump(save_list, fp)

        else:
            for index, id in enumerate(id_list):
                name = id.split("/")[-1]

                output = results[0][index, ...]
                output_path = opj(config.results_path, 'output_')

                np.save(output_path + name, output.cpu().detach().numpy())

                if (config.network == "VanillaVAE") or (config.network == "SpatialVAE"):
                    mu = results[2][index, ...]
                    logvar = results[3][index, ...]
                    mu_path = opj(config.results_path, 'mu_')
                    logvar_path = opj(config.results_path, 'logvar_')
                    np.save(mu_path+name, mu.cpu().detach().numpy())
                    np.save(logvar_path+name, logvar.cpu().detach().numpy())


    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                            gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                return optims, scheds
        except:
            return optims