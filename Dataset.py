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


def image_in_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    cnt = 2
    while cnt > 0 and random.random() < 0.95:
        # Shape the block
        block_noise_size_x = random.randint(img_rows//12, img_rows//6)
        block_noise_size_y = random.randint(img_cols//12, img_cols//6)
        block_noise_size_z = random.randint(img_deps//12, img_deps//6)
        # Place the block
        noise_x = random.randint(6, img_rows-block_noise_size_x-6)
        noise_y = random.randint(6, img_cols-block_noise_size_y-6)
        noise_z = random.randint(6, img_deps-block_noise_size_z-6)
        x[:,
          noise_x:noise_x+block_noise_size_x,
          noise_y:noise_y+block_noise_size_y,
          noise_z:noise_z+block_noise_size_z] = np.random.rand(block_noise_size_x,
                                                               block_noise_size_y,
                                                               block_noise_size_z, ) * 1.0
        cnt -= 1
    return x

def test_print(img, aug_img):
    plt.figure()
    plt.imshow(img[0, :, :, 20], cmap='gray')
    plt.show()
    plt.close()

    plt.figure()
    plt.imshow(aug_img[0, :, :, 20], cmap='gray')
    plt.show()
    plt.close()

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def nonlinear_transformation(x, prob=0.9):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def data_augmentation(img):
    # Basic augmentation by applying a non linear transformation to the value of each voxel.
    nonlinear = nonlinear_transformation(img, prob=0.5).astype(np.float32)
    inpainting = image_in_painting(img)
    # Do not change axis. Output must be (64,64,80,64)
    return nonlinear, inpainting


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
                if self.type == "test":
                    b0_mask = np.load(config.img_path_uka+'/Test/b0_brainmask'+mask_id)
                else:
                    b0_mask = np.load(config.img_path_uka + '/Train/b0_brainmask_' + mask_id)

                patch = np.pad(patch, [(0, 0), (1, 1), (1, 1), (1, 1)], mode='constant', constant_values=0)
                b0_mask = np.pad(b0_mask, [(1, 1), (1, 1), (1, 1)], mode='constant', constant_values=0)

                if 0:
                    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
                    ax[0].imshow(patch[0, :, :, 30], cmap='gray')
                    ax[0].axis('off')
                    ax[1].imshow(b0_mask[:, :, 30], cmap='gray')
                    ax[1].axis('off')
                    plt.tight_layout()
                    plt.show()
                    plt.close(fig)

                for idx, x in np.ndenumerate(patch[0, ...]):
                    # Only use voxel from brain for training/validation/testing
                    if b0_mask[idx[0], idx[1], idx[2]] == 0:
                        continue

                    if config.network == "CNNVoxelVAE":
                        voxel_input = patch[:, (idx[0]-1):(idx[0]+2), (idx[1]-1):(idx[1]+2), (idx[2]-1):(idx[2]+2)]
                        voxel_target = patch[:, (idx[0]-1):(idx[0]+2), (idx[1]-1):(idx[1]+2), (idx[2]-1):(idx[2]+2)]

                        #if voxel_input.shape != (64,3,3,3):
                        #    a = 1subject.split("/")[-1]
                    else:
                        voxel_input = patch[:, idx[0], idx[1], idx[2]]
                        voxel_target = patch[:, idx[0], idx[1], idx[2]]

                    # Padding the arrays shifts the coordinates
                    coordinates = (idx[0]-1, idx[1]-1, idx[2]-1)
                    self.patches.append((voxel_input, voxel_target, subject, coordinates))

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
                    "coordinates": self.patches[idx][3]}
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