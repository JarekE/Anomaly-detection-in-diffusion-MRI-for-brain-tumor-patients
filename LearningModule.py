# Here I will define all my deep learning
import torch
from pytorch_lightning.core.lightning import LightningModule
import config
from Models.VAE import VanillaVAE
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as opj
from Models.UNet import UNet3d

import config


class LearningModule(LightningModule):

    def __init__(self):
        super(LearningModule, self).__init__()

        # Choose from available networks
        if config.network == "VanillaVAE":
            self.model = VanillaVAE(in_channels=64, latent_dim=config.latent_dim)
            self.params = config.vanilla_params
        if config.network == "UNet":
            self.model = UNet3d(in_channels=64)
            self.params = config.unet_params
        else:
            raise ValueError('You chose a network that is not available atm.')


    def forward(self, z):
        y = self.model(z)

        return y

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        target = batch['target']
        input = batch['input']

        results = self.forward(input)

        results.append(target)

        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'])

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        target = batch['target']
        input = batch['input']

        results = self.forward(input)

        results.append(target)

        val_loss = self.model.loss_function(*results,
                                            M_N=1.0)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        pass

    def test_step(self, batch, batch_idx):
        #labels = batch['target']
        real_img = batch['input']
        id_list = batch['id']

        results = self.forward(real_img)

        for index, id in enumerate(id_list):
            name = id.split("/")[-1]

            output = results[0][index, ...]
            output_path = opj(config.results_path, 'output_')
            np.save(output_path + name, output.cpu().detach().numpy())

            if config.network == "VanillaVAE":
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