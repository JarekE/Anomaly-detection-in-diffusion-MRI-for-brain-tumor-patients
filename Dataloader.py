# Load my data into the pipeline
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from Dataset import UKADataset
import config

class DataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        ...

    def train_dataloader(self):
        self.train_dataset = UKADataset(type="training")
        return DataLoader(self.train_dataset, batch_size=config.batch_size,
                                     shuffle=False, num_workers=0)

    def val_dataloader(self):
        self.val_dataset = UKADataset(type="validation")
        return DataLoader(self.val_dataset, batch_size=config.batch_size,
                                         shuffle=False, num_workers=0)

    def test_dataloader(self):
        self.test_dataset = UKADataset(type="test")
        return DataLoader(self.test_dataset, batch_size=config.batch_size,
                          shuffle=False, num_workers=0)