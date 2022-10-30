from pytorch_lightning.core.lightning import LightningModule
from Models.VAE import VanillaVAE
from Models.RecDiscNet import RecDisc
from torch import optim
import numpy as np
from os.path import join as opj
from Models.UNet import UNet3d 
import config
from DataProcessing.data_visualization import show_RecDisc
import matplotlib.pyplot as plt


class LearningModule(LightningModule):

    def __init__(self):
        super(LearningModule, self).__init__()

        if config.network == "VanillaVAE":
            self.model = VanillaVAE(in_channels=64, latent_dim=config.latent_dim)
            self.params = config.vanilla_params
        elif config.network == "UNet":
            self.model = UNet3d(in_channels=64)
            self.params = config.unet_params
        elif config.network == "RecDisc":
            self.model = RecDisc(in_channels=64, in_channels_unet=64*2)
            self.params = config.rec_disc_params
        else:
            raise ValueError('You chose a network that is not available at the moment.: '+config.network)

    def forward(self, input):
        y = self.model(input)
        return y

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        target = batch['target']
        input = batch['input']

        if config.network == "RecDisc":
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

        if config.network == "RecDisc":
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
            val_loss = self.model.loss_function(*results)
            if config.network == "RecDisc" and ((self.trainer.current_epoch < 30) or ((self.trainer.current_epoch/50).is_integer())) and batch_idx == 0:
                show_RecDisc(results[3], results[2], results[4], results[0], results[5], results[1])
        else:
            results = self.forward(input)
            results.append(target)

            if self.trainer.current_epoch < 1 and batch_idx == 1:
                fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))
                ax[0].imshow(input.detach().cpu().numpy()[1, 30, :, :, 40], cmap='gray')
                ax[0].axis('off')
                ax[0].title.set_text("Input")
                ax[1].imshow(results[0].detach().cpu().numpy()[1, 30, :, :, 40], cmap='gray')
                ax[1].axis('off')
                ax[1].title.set_text("Out")
                ax[2].imshow(np.mean(input.detach().cpu().numpy(), axis=1)[1, :, :, 40], cmap='gray')
                ax[2].axis('off')
                ax[2].title.set_text("Input Mean")
                ax[3].imshow(np.mean(results[0].detach().cpu().numpy(), axis=1)[1, :, :, 40], cmap='gray')
                ax[3].axis('off')
                ax[3].title.set_text("Output Mean")
                plt.tight_layout()
                plt.show()
                plt.close(fig)

            val_loss = self.model.loss_function(*results, M_N=self.params['kld_weight'])

        print(val_loss.items())
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def test_step(self, batch, batch_idx):
        real_img = batch['input']
        id_list = batch['id']

        results = self.forward(real_img)

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

        for index, id in enumerate(id_list):
            name = id.split("/")[-1]

            output = results[0][index, ...]
            output_path = opj(config.results_path, 'output_')
            np.save(output_path + name, output.cpu().detach().numpy())

            if config.network == "RecDisc":
                rec = results[1][index, ...]
                rec_path = opj(config.results_path, 'rec_')
                np.save(rec_path + name, rec.cpu().detach().numpy())

            if config.network == "VanillaVAE":
                mu = results[2][index, ...]
                logvar = results[3][index, ...]
                mu_path = opj(config.results_path, 'mu_')
                logvar_path = opj(config.results_path, 'logvar_')
                np.save(mu_path+name, mu.cpu().detach().numpy())
                np.save(logvar_path+name, logvar.cpu().detach().numpy())

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