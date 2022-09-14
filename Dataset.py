# Here I will preprocess my Dataset!

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from scipy.special import comb
import matplotlib.pyplot as plt
from LearningModule import LearningModule
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

            if (config.network == "VoxelVAE") or (config.network == "CNNVoxelVAE"):
                patch = np.load(subject)
                # Can be used for different input and target
                test_patch = patch.copy()
                if self.type == "test":
                    mask = np.load(config.img_path_uka+'/Test/brainmask_withoutCSF'+mask_id)
                    groundtruth = np.load(config.img_path_uka+'/Test/mask'+mask_id)
                else:
                    mask = np.load(config.img_path_uka + '/Train/brainmask_withoutCSF' + mask_id)
                    patch, groundtruth, _ = anomaly_generation(patch, mask)
                    groundtruth = np.squeeze(groundtruth)
                    print("Anomaly is in image: ", not(np.array_equal(test_patch, groundtruth)))

                groundtruth = np.pad(groundtruth, [(1, 1), (1, 1), (1, 1)], mode='constant', constant_values=0)
                patch = np.pad(patch, [(0, 0), (1, 1), (1, 1), (1, 1)], mode='constant', constant_values=0)
                mask = np.pad(mask, [(1, 1), (1, 1), (1, 1)], mode='constant', constant_values=0)

                n = 0
                for idx, x in np.ndenumerate(patch[0, ...]):
                    # Only use voxel from wm or gm for training/validation/testing
                    if mask[idx[0], idx[1], idx[2]] == 0:
                        n = n+1
                        continue

                    if config.network == "CNNVoxelVAE":
                        voxel_input = patch[:, (idx[0]-1):(idx[0]+2), (idx[1]-1):(idx[1]+2), (idx[2]-1):(idx[2]+2)]
                        voxel_target = patch[:, (idx[0]-1):(idx[0]+2), (idx[1]-1):(idx[1]+2), (idx[2]-1):(idx[2]+2)]
                        if groundtruth[idx[0], idx[1], idx[2]] == 0:
                            vector_class = 0
                        else:
                            vector_class = 1
                    else:
                        voxel_input = patch[:, idx[0], idx[1], idx[2]]
                        voxel_target = patch[:, idx[0], idx[1], idx[2]]
                        # Add coordiantes to vector (normalised over length of image)
                        voxel_input = np.append(voxel_input, np.float32([[(idx[0]-1)/64], [(idx[1]-1)/80], [(idx[2]-1)/64]]))
                        voxel_target = np.append(voxel_target, np.float32([[(idx[0]-1)/64], [(idx[1]-1)/80], [(idx[2]-1)/64]]))
                        # Add class to vector
                        if groundtruth[idx[0], idx[1], idx[2]] == 0:
                            vector_class = 0
                        else:
                            vector_class = 1

                    # Padding the arrays shifts the coordinates
                    coordinates = (idx[0]-1, idx[1]-1, idx[2]-1)

                    # Data augmentation of anomaly train data
                    if self.type != "test" and groundtruth[idx[0], idx[1], idx[2]] == 1:
                        # Every input voxel is repeated 64 times
                        for i in range(64):
                            self.patches.append((voxel_input, voxel_target, subject, coordinates, vector_class))
                    else:
                        self.patches.append((voxel_input, voxel_target, subject, coordinates, vector_class))
                random.shuffle(self.patches)

            else:
                patch = np.load(subject)
                target = patch.copy()
                if self.type == "test":
                    mask_without_csf = np.load('/work/scratch/ecke/Masterarbeit/Data/Test/brainmask_withoutCSF'+ mask_id)
                else:
                    mask_without_csf = np.load('/work/scratch/ecke/Masterarbeit/Data/Train/brainmask_withoutCSF'+ mask_id)

                # Anomaly generation
                if (config.network == "RecDiscUnet") or (config.network == "RecDisc"):
                    if self.type == "test":
                        # This mask is not in use atm, but still necessary
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

        if (config.network == "VoxelVAE") or (config.network == "CNNVoxelVAE"):
            return {"input": self.patches[idx][0],
                    "target": self.patches[idx][1],
                    "id": self.patches[idx][2],
                    "coordinates": self.patches[idx][3],
                    "vector_class": self.patches[idx][4]}
        elif (config.network == "RecDiscUnet") or (config.network == "RecDisc"):
            return {"input": self.patches[idx][0],
                    "target": self.patches[idx][1],
                    "id": self.patches[idx][2],
                    "print": self.patches[idx][3],
                    "raw_input": self.patches[idx][4]}
        else:
            return {"input": self.patches[idx][0],
                    "target": self.patches[idx][1],
                    "id": self.patches[idx][2]}