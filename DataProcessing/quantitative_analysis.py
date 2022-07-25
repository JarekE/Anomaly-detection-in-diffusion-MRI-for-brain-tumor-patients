"""

Load the trained data for a quantitative analysis

- Load n different results for the same model (e.g. size of latent space)

- Calculate for each image in the result-dataset the roc curve and the AUC (metric)
    - Use only the wm data
- Average the results in one dataset

- Repeat for n result

- Print results in one diagram


Addition:
- find best threshold + calculate f1 (compare to approach 3)
- same with PR curve and a metric

"""
from glob import glob
from os.path import join as opj
import os
import shutil
from glob import glob
from os.path import join as opj
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import montage
from scipy import ndimage
from skimage import filters
import pickle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

"""
Add all results to compare
"""
result_list = []


result1 = "VanillaVAE-epoch=150-val_loss=0.02-max_epochs=300-latent_dim=16.ckpt"
result_list.append(result1)


# All necessary paths and information
PATH = '/work/scratch/ecke/Masterarbeit/Results'
DATA_PATH = '/work/scratch/ecke/Masterarbeit/Data/Test'


def roc(map, brainmask_NoCSF, tumormask):
    # Change tumor mask (target) to a binary mask
    tumormask[tumormask > 0] = 1

    # Choose only values in the WM section
    map = np.where(brainmask_NoCSF == 1, map, 0)
    tumormask = np.where(brainmask_NoCSF == 1, tumormask, 0)
    map = map.flatten()
    tumormask = tumormask.flatten()
    tumormask = tumormask[map != 0]
    map = map[map != 0]

    ns_probs = [0 for _ in range(len(tumormask))]
    # calculate scores
    ns_auc = roc_auc_score(tumormask, ns_probs)
    lr_auc = roc_auc_score(tumormask, map)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(tumormask, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(tumormask, map)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Model')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

    f1 = 100

    return f1, lr_auc, lr_fpr, lr_tpr


for item in result_list:
    map_list = glob(opj(PATH, item, "map*"))
    map_list.sort()
    brainmask_NoCSF_list = glob(opj(DATA_PATH, "brainmask_withoutCSF*"))
    brainmask_NoCSF_list.sort()
    tumor_list = glob(opj(DATA_PATH, "mask*"))
    tumor_list.sort()

    for i in range(len(map_list)):
        map = np.load(map_list[i])
        brainmask_NoCSF = np.load(brainmask_NoCSF_list[i])
        tumor = np.load(tumor_list[i])
        f1, auc, fpr, tpr = roc(map, brainmask_NoCSF, tumor)

