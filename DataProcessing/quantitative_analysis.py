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

from statistics import mean
from glob import glob
from os.path import join as opj
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from operator import itemgetter
import statistics

"""
Add all results to compare

Example Baseline: 5x5 results = 25
"""
result_list = []
list = []


result1 = 'VanillaVAE-epoch=226-val_loss=0.02-max_epochs=300-latent_dim=1.ckpt'
result_list.append(result1)
result2 = 'VanillaVAE-epoch=209-val_loss=0.02-max_epochs=300-latent_dim=2.ckpt'
result_list.append(result2)
result4 = 'VanillaVAE-epoch=217-val_loss=0.02-max_epochs=300-latent_dim=4.ckpt'
result_list.append(result4)
result8 = 'VanillaVAE-epoch=194-val_loss=0.02-max_epochs=300-latent_dim=8.ckpt'
result_list.append(result8)
result16 = 'VanillaVAE-epoch=150-val_loss=0.02-max_epochs=300-latent_dim=16.ckpt'
result_list.append(result16)
result32 = 'VanillaVAE-epoch=282-val_loss=0.02-max_epochs=300-latent_dim=32.ckpt'
result_list.append(result32)
result64 = 'VanillaVAE-epoch=84-val_loss=0.02-max_epochs=300-latent_dim=64.ckpt'
result_list.append(result64)
result128 = 'VanillaVAE-epoch=167-val_loss=0.02-max_epochs=300-latent_dim=128.ckpt'
result_list.append(result128)
result256 = 'VanillaVAE-epoch=182-val_loss=0.02-max_epochs=300-latent_dim=256.ckpt'
result_list.append(result256)
result512 = 'VanillaVAE-epoch=182-val_loss=0.02-max_epochs=300-latent_dim=512.ckpt'
result_list.append(result512)


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

    # ROC AUC + f1
    ns_probs = [0 for _ in range(len(tumormask))]
    ns_auc = roc_auc_score(tumormask, ns_probs)
    auc = roc_auc_score(tumormask, map)
    ns_fpr, ns_tpr, _ = roc_curve(tumormask, ns_probs)
    fpr, tpr, _ = roc_curve(tumormask, map)

    # Here: Decide for best threshold (must be the same arguments for all) and afterwards calculate the f1 score
    f1 = 0

    if 0:
        # summarize scores
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Model: ROC AUC=%.3f' % (auc))
        # plot the roc curve for the model
        pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        pyplot.plot(fpr, tpr, marker='.', label='Model')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()

    # Bring fpr and tpr to 40 evenly spaced values
    idx = np.round(np.linspace(0, len(fpr) - 1, 40)).astype(int)
    fpr = fpr[idx]
    tpr = tpr[idx]

    return f1, auc, fpr, tpr


def print_curve(list):
    #marker = ["v", "^", "<", ">", "8", "s", "p", "*", ".", "D"]
    for i in range(len(list)):
        print('Model' + str(list[i][4]) + 'Epochs' + str(list[i][5]) + ': ROC AUC=%.3f' % (list[i][1]))
        pyplot.plot(list[i][2], list[i][3], marker=".",
                    label='Model_' + str(list[i][4]) + '_Epochs_' + str(list[i][5]) + ' AUC: ' + str(list[i][1].astype(np.float32)))

    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.show()


for item in result_list:
    map_list = glob(opj(PATH, item, "map*"))
    map_list.sort()
    brainmask_NoCSF_list = glob(opj(DATA_PATH, "brainmask_withoutCSF*"))
    brainmask_NoCSF_list.sort()
    tumor_list = glob(opj(DATA_PATH, "mask*"))
    tumor_list.sort()
    f1_list, auc_list, fpr_list, tpr_list = [], [], [], []
    name_epochs = item[item.find('max_epochs=') + len('max_epochs='):item.rfind('-latent_dim')]
    name_latentdim = item[item.find('latent_dim=') + len('latent_dim='):item.rfind('.ckpt')]

    for i in range(len(map_list)):
        map = np.load(map_list[i])
        brainmask_NoCSF = np.load(brainmask_NoCSF_list[i])
        tumor = np.load(tumor_list[i])
        f1, auc, fpr, tpr = roc(map, brainmask_NoCSF, tumor)
        f1_list.append(f1)
        auc_list.append(auc)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    """
    TODO
    Calculate sd of results and safe everything in a excel sheet
    """
    f1_average = mean(f1_list)
    auc_average = mean(auc_list)
    st_dev = statistics.pstdev(auc_list)
    fpr_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*fpr_list)]
    tpr_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*tpr_list)]

    locals()['latent_space' + name_latentdim+'epochs' + name_epochs] = [f1_average, auc_average, fpr_average, tpr_average, name_latentdim, name_epochs, st_dev]
    list.append(locals()['latent_space' + name_latentdim + 'epochs'+name_epochs])

"""
TODO
Write function to sort list and only print 3 best and worst curves
"""
list = sorted(list, key=itemgetter(1))
print_curve(list[0:3]+list[-3:-1])