import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from DataProcessing.anomaly_generation import anomaly_generation
import config
from config import uka_subjects


class UKADataset(Dataset):
    def __init__(self, type):

        self.type = type
        self.subject_ids = uka_subjects[type]
        self.patches = []

        print("\n Loading data...")
        for subject in tqdm(self.subject_ids):
            mask_id = subject.split("/")[-1]
            patch = np.load(subject)
            target = patch.copy()
            if self.type == "test":
                mask_without_csf = np.load('/work/scratch/ecke/Masterarbeit/Data/Test/brainmask_withoutCSF'+ mask_id)
            else:
                mask_without_csf = np.load('/work/scratch/ecke/Masterarbeit/Data/Train/brainmask_withoutCSF'+ mask_id)

            # Anomaly generation
            if config.network == "RecDisc":
                if self.type == "test":
                    mask_without_csf = mask_without_csf[None, :, :, :]
                    self.patches.append((patch, mask_without_csf, subject, 30, patch))
                else:
                    for i in range(4):
                        anomaly_x, reconstructive_map, z = anomaly_generation(patch, mask_without_csf)
                        self.patches.append((anomaly_x, reconstructive_map, subject, z, patch))
                    random.shuffle(self.patches)
            else:
                self.patches.append((patch, target, subject))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        if config.network == "RecDisc":
            return {"input": self.patches[idx][0],
                    "target": self.patches[idx][1],
                    "id": self.patches[idx][2],
                    "print": self.patches[idx][3],
                    "raw_input": self.patches[idx][4]}
        else:
            return {"input": self.patches[idx][0],
                    "target": self.patches[idx][1],
                    "id": self.patches[idx][2]}