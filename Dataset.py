# Here I will preprocess my Dataset!

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from config import uka_subjects


class UKADataset(Dataset):
    def __init__(self, type):

        self.type = type
        self.subject_ids = uka_subjects[type]
        self.patches = []

        print("\n Loading data...")
        for subject in tqdm(self.subject_ids):

            patch = np.load(subject)
            target = patch
            self.patches.append((patch, target, subject))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return {"input": self.patches[idx][0],
                "target": self.patches[idx][1],
                "id": self.patches[idx][2]}