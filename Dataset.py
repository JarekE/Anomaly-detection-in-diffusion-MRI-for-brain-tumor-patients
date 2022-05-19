# Here I will preprocess my Dataset!

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from scipy.special import comb
import matplotlib.pyplot as plt

import config
from config import uka_subjects

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
    nonlinear = nonlinear_transformation(img, prob=1).astype(np.float32)
    # Do not change axis. Output must be (64,64,80,64)
    return nonlinear


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

            if (type == "training") and (config.augmentation == True):
                aug_patch = data_augmentation(patch)
                self.patches.append((aug_patch, target, subject))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return {"input": self.patches[idx][0],
                "target": self.patches[idx][1],
                "id": self.patches[idx][2]}