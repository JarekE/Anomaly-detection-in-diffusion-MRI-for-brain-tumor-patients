import pandas as pd
from glob import glob
from os.path import join as opj
import numpy as np
from skimage import filters
from sklearn.metrics import roc_curve, f1_score, roc_auc_score
import statistics
from statistics import mean
import matplotlib.pyplot as plt

#####################################
#
#
# Code snippets for further processing or an insight of data
#
#
#####################################

# Save results in an excel file
def results_to_excel():
    load_run = "/work/scratch/ecke/Masterarbeit/Results_RecDiscNet/Results_Architecture/Run10"
    load_list = glob(opj(load_run, "*"))
    df = pd.DataFrame(columns=['Model','FilterRec','ActDisc','ActRec','ReLu','ValLoss'])
    list_of_lists, model, filter, disc, rec, relu, loss = [], [], [], [], [], [], []

    for item in load_list:
        item = item.split("/")[-1]
        model.append(item[:item.rfind('-epoch')])
        filter.append(item[item.find('f=') + len('f='):item.rfind('-ar')])
        disc.append(item[item.find('a=') + len('a='):item.rfind('-leaky')])
        rec.append(item[item.find('ar=') + len('ar='):item.rfind('-a')])
        relu.append(item[item.find('leaky=') + len('leaky='):item.rfind('.ckpt')])
        loss.append(item[item.find('val_loss=') + len('val_loss='):item.rfind('-r')])

    list_of_lists = list(zip(model, filter, disc, rec, relu, loss))
    df = df.append(pd.DataFrame(list_of_lists, columns=df.columns))
    df.to_csv(opj(load_run, 'run.csv'), index=False, header=True)
    return

# Get a first insight into the final data
def first_insight():
    load = "/work/scratch/ecke/Masterarbeit/Results/RecDisc-epoch=203-val_loss=0.15-r=1-an=Mix-d=Half.ckpt"
    DATA_PATH = '/work/scratch/ecke/Masterarbeit/Data/Test'
    load_list = glob(opj(load, "map*"))
    load_list.sort()
    f1_list, f1_otsu_list, tpr_list, fpr_list, auc_list = [], [], [], [], []

    brainmask_NoCSF_list = glob(opj(DATA_PATH, "brainmask_withoutCSF*"))
    brainmask_NoCSF_list.sort()
    tumor_list = glob(opj(DATA_PATH, "mask*"))
    tumor_list.sort()

    for i in range(len(load_list)):
        map = np.load(load_list[i])
        mask_id = load_list[i].split("/")[-1]
        brainmask_NoCSF = np.load(brainmask_NoCSF_list[i])
        tumormask = np.load(tumor_list[i])
        tumormask[tumormask > 0] = 1

        unsupervised_threshold = 0.5
        map = np.where(brainmask_NoCSF == 1, map, 0)
        tumormask = np.where(brainmask_NoCSF == 1, tumormask, 0)
        map_flatten = map.flatten()
        tumormask_flatten = tumormask.flatten()

        tumormask_flatten = tumormask_flatten[map_flatten != 0]
        map_flatten = map_flatten[map_flatten != 0]

        otsu_threhold = filters.threshold_otsu(map_flatten)

        # 0.5 Threshold
        f1 = f1_score(tumormask_flatten, np.where(map_flatten >= unsupervised_threshold, 1, 0))
        f1_list.append(f1)
        # Otsu Threshold
        f1_otsu = f1_score(tumormask_flatten, np.where(map_flatten >= otsu_threhold, 1, 0))
        f1_otsu_list.append((f1_otsu))
        print("Image: ", mask_id, " F1-Score: ", f1, " F1-Score Otsu: ", f1_otsu)

        #AUC
        auc = roc_auc_score(tumormask_flatten, map_flatten)
        fpr, tpr, _ = roc_curve(tumormask_flatten, map_flatten)
        idx = np.round(np.linspace(0, len(fpr) - 1, 40)).astype(int)
        fpr = fpr[idx]
        tpr = tpr[idx]
        auc_list.append(auc)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    f1_mean = mean(f1_list)
    st_dev_f1 = statistics.pstdev(f1_list)
    f1_otsu_mean = mean(f1_otsu_list)
    st_dev_otsu_f1 = statistics.pstdev(f1_otsu_list)
    auc_average = mean(auc_list)
    st_dev = statistics.pstdev(auc_list)
    fpr_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*fpr_list)]
    tpr_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*tpr_list)]

    print("F1-Mean: ", f1_mean, "+-", st_dev_f1, " F1-Mean Otsu: ", f1_otsu_mean, " +-", st_dev_otsu_f1)
    plt.plot(fpr_average, tpr_average, marker=".", label='AUC: ' + str(auc_average) + '\u00B1' + str(st_dev))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()