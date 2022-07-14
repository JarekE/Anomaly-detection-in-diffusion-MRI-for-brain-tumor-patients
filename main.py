# Call training and testing. In both cases use /Results for the results.
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from LearningModule import LearningModule
from Dataloader import DataModule
import config
import postprocessing
import os
import shutil


def main():

    # Reset
    torch.cuda.empty_cache()
    seed_everything(42)
    print(config.args)

    # Test
    if config.mode == "test":

        if os.path.exists("logs/DataDropOff"):
            shutil.rmtree("logs/DataDropOff")
        os.mkdir("logs/DataDropOff")

        path = postprocessing.prepare_results()
        model = LearningModule.load_from_checkpoint(path)
        dataloader = DataModule()
        trainer = pl.Trainer(gpus=1)
        trainer.test(model, test_dataloaders=dataloader.test_dataloader())
        postprocessing.processing()
        quit()

    # Load
    model = LearningModule()
    dataloader = DataModule()

    # Log and Call
    logger = CSVLogger(save_dir=config.log_dir_logger,
                                  name=config.network)
    callbacks = ModelCheckpoint(save_top_k=1,
                                dirpath=config.log_dir,
                                monitor="val_loss",
                                filename=config.file_name,
                                save_last=True)

    # Train
    trainer = pl.Trainer(gpus=1,
                         max_epochs=config.epochs,
                         deterministic=True,
                         logger=logger,
                         callbacks=[callbacks],
                         log_every_n_steps=10)

    trainer.fit(model, datamodule=dataloader)

if __name__ == "__main__":
    main()
