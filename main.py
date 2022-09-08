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
    #seed_everything(42)
    print(config.args)

    # Test
    if config.mode == "test":

        if os.path.exists(config.data_drop_off): 
            shutil.rmtree(config.data_drop_off)
        os.makedirs(config.data_drop_off)

        path = postprocessing.prepare_results()
        model = LearningModule.load_from_checkpoint(path)
        dataloader = DataModule()
        trainer = pl.Trainer(gpus=1)
        trainer.test(model, test_dataloaders=dataloader.test_dataloader())
        postprocessing.processing()

        shutil.rmtree(config.data_drop_off)
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

    # Automate testing after each run (remember: analysis is still extern and testing can also be run extern)
    where_is_checkpoint = callbacks.best_model_path
    print(where_is_checkpoint)
    test_id = where_is_checkpoint.split("/")[-1]
    config.create_id(test_id)

    if os.path.exists(config.data_drop_off):
        shutil.rmtree(config.data_drop_off)
    os.makedirs(config.data_drop_off)

    path = postprocessing.prepare_results()
    model = LearningModule.load_from_checkpoint(path)
    dataloader = DataModule()
    trainer = pl.Trainer(gpus=1)
    trainer.test(model, test_dataloaders=dataloader.test_dataloader())
    postprocessing.processing()

    shutil.rmtree(config.data_drop_off)


if __name__ == "__main__":
    main()