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
import matplotlib.pyplot as plt


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


    def forward(self, input):
        y = self.model(input)

        return y

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        target = batch['target']
        input = batch['input']

        if (config.network == "RecDisc") or (config.network == "RecDiscUnet"):
            print_value = batch['print']
            raw_input = batch['raw_input']

            if self.trainer.current_epoch < 1 and batch_idx == 1:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
                ax[0].imshow(input.detach().cpu().numpy()[2, 30, :, :, print_value.detach().cpu().numpy()[2]], cmap='gray')
                ax[0].axis('off')
                ax[0].title.set_text("Anomaly")
                ax[1].imshow(raw_input.detach().cpu().numpy()[2, 30, :, :, print_value.detach().cpu().numpy()[2]],
                             cmap='gray')
                ax[1].axis('off')
                ax[1].title.set_text("Raw Input")
                plt.tight_layout()
                plt.show()
                plt.close(fig)

            results = self.forward(input)
            results.extend([raw_input, target, print_value])
            # [disc, rec, anomaly_data, input, rec_map, z]
            train_loss = self.model.loss_function(*results)
        else:
            results = self.forward(input)
            results.append(target)
            train_loss = self.model.loss_function(*results, M_N=self.params['kld_weight'])

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        target = batch['target']
        input = batch['input']

        if config.network == "RecDisc" or (config.network == "RecDiscUnet"):
            print_value = batch['print']
            raw_input = batch['raw_input']

            if self.trainer.current_epoch < 1 and batch_idx == 1:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
                ax[0].imshow(input.detach().cpu().numpy()[2, 30, :, :, print_value.detach().cpu().numpy()[2]], cmap='gray')
                ax[0].axis('off')
                ax[0].title.set_text("Anomaly")
                ax[1].imshow(raw_input.detach().cpu().numpy()[2, 30, :, :, print_value.detach().cpu().numpy()[2]], cmap='gray')
                ax[1].axis('off')
                ax[1].title.set_text("Raw Input")
                plt.tight_layout()
                plt.show()
                plt.close(fig)

            results = self.forward(input)
            results.extend([raw_input, target, print_value])
            # [disc, rec, anomaly_data, raw_input, rec_map, z]
            # [disc, rec, anomaly_data, mu, log_var, raw_input, rec_map, z]
            val_loss = self.model.loss_function(*results)
            if config.network == "RecDisc" and ((self.trainer.current_epoch < 30) or ((self.trainer.current_epoch/50).is_integer())) and batch_idx == 0:
                #show_RecDisc(results[5], results[2], results[6], results[0], results[7], results[1])
                show_RecDisc(results[3], results[2], results[4], results[0], results[5], results[1])
            if config.network == "RecDiscUnet" and ((self.trainer.current_epoch < 30) or ((self.trainer.current_epoch/50).is_integer())) and batch_idx == 0:
                # input, input_anomaly, reconstructive_map, results, z, reconstruction
                show_RecDisc(results[3], results[2], results[4], results[0], results[5], results[1])
        else:
            results = self.forward(input)
            results.append(target)
            val_loss = self.model.loss_function(*results, M_N=self.params['kld_weight'])

        print(val_loss.items())
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def test_step(self, batch, batch_idx):
        #labels = batch['target']
        real_img = batch['input']
        id_list = batch['id']
        # mask = batch['mask_withoutCSF']
        if batch_idx == 1:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
            ax[0, 0].imshow(real_img.detach().cpu().numpy()[2, 30, :, :, 24], cmap='gray')
            ax[0, 0].axis('off')
            ax[0, 0].title.set_text("Input")
            ax[0, 1].imshow(real_img.detach().cpu().numpy()[2, 30, :, :, 32], cmap='gray')
            ax[0, 1].axis('off')
            ax[0, 1].title.set_text("Input")
            ax[1, 0].imshow(real_img.detach().cpu().numpy()[2, 30, :, :, 40], cmap='gray')
            ax[1, 0].axis('off')
            ax[1, 0].title.set_text("Input")
            ax[1, 1].imshow(real_img.detach().cpu().numpy()[2, 30, :, :, 48], cmap='gray')
            ax[1, 1].axis('off')
            ax[1, 1].title.set_text("Input")
            plt.tight_layout()
            plt.show()
            plt.close(fig)

        results = self.forward(real_img)

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