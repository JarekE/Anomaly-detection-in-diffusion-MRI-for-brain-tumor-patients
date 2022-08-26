from statistics import mean
from glob import glob
from os.path import join as opj
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import roc_curve, f1_score, roc_auc_score, accuracy_score
from operator import itemgetter
import statistics
import pandas as pd
from scipy.ndimage.morphology import binary_opening


# All necessary paths and information
DATA_PATH = '/work/scratch/ecke/Masterarbeit/Data/Test'


def calculate_metrics(threshold, mask, map, opening):

    map = np.where(map >= threshold, 1, 0)
    f1 = f1_score(mask, map)
    f1_opening = f1_score(mask, opening)
    acc = accuracy_score(mask, map)
    acc_opening = accuracy_score(mask, opening)

    return f1, acc, f1_opening, acc_opening

def roc(map, brainmask_NoCSF, tumormask):
    # Change tumor mask (target) to a binary mask
    tumormask[tumormask > 0] = 1

    # Unsupervised threshold!
    unsupervised_threshold = 0.5

    # Choose only values in the WM section
    map = np.where(brainmask_NoCSF == 1, map, 0)
    map_opening = binary_opening(np.where(map >= unsupervised_threshold, 1, 0), structure=np.ones((3, 3, 3)))
    tumormask = np.where(brainmask_NoCSF == 1, tumormask, 0)
    map_flatten = map.flatten()
    tumormask_flatten = tumormask.flatten()
    map_opening_flatten = map_opening.flatten()

    tumormask_flatten = tumormask_flatten[map_flatten != 0]
    map_opening_flatten = map_opening_flatten[map_flatten != 0]
    map_flatten = map_flatten[map_flatten != 0]

    # ROC AUC
    auc = roc_auc_score(tumormask_flatten, map_flatten)
    fpr, tpr, threshold = roc_curve(tumormask_flatten, map_flatten)

    # Find optimal threshold (this is not undsupervised, therefore not used to produce quantitative results)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]

    # Metrics
    f1, acc, f1_opening, acc_opening = calculate_metrics(unsupervised_threshold, tumormask_flatten, map_flatten, map_opening_flatten)

    # Bring fpr and tpr to 40 evenly spaced values
    idx = np.round(np.linspace(0, len(fpr) - 1, 40)).astype(int)
    fpr = fpr[idx]
    tpr = tpr[idx]

    return f1, auc, fpr, tpr, acc, optimal_threshold, f1_opening, acc_opening


def print_curve(list):
    #marker = ["v", "^", "<", ">", "8", "s", "p", "*", ".", "D"]
    for i in range(len(list)):
        print('Model' + str(list[i][4]) + 'Epochs' + str(list[i][5]) + ': ROC AUC=%.3f' % (list[i][1]))
        pyplot.plot(list[i][2], list[i][3], marker=".",
                    label='Model_' + str(list[i][4]) + '_Epochs_' + str(list[i][5]) + ' AUC: ' + str(np.around(list[i][1], 4)) + '\u00B1' + str(round(list[i][6], 3)))

    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.savefig(opj(analysis_results, 'curve.png'), bbox_inches='tight', transparent = True)
    pyplot.show()


    # zoomed
    for i in range(len(list)):
        print('Model' + str(list[i][4]) + 'Epochs' + str(list[i][5]) + ': ROC AUC=%.3f' % (list[i][1]))
        pyplot.plot(list[i][2], list[i][3], marker=".",
                    label='Model_' + str(list[i][4]) + '_' + str(list[i][5]) + ' AUC: ' + str(np.around(list[i][1], 4)) + ' \u00B1 ' + str(round(list[i][6], 3)))

    pyplot.xlabel('False Positive Rate (zoomed in)')
    pyplot.ylabel('True Positive Rate (zoomed in)')
    pyplot.axis([0.1, 0.5, 0.3, 0.8])
    pyplot.legend()
    pyplot.savefig(opj(analysis_results, 'curve_zoomed.png'), bbox_inches='tight', transparent = True)
    pyplot.show()


def average_results(list1, list2, list3, number):

    for i in range(len(list1)):
        list1[i][0] = (list1[i][0] + list2[i][0] + list3[i][0]) / number
        list1[i][1] = (list1[i][1] + list2[i][1] + list3[i][1]) / number
        list1[i][2] = [(g + h + z) / number for g, h, z in zip(list1[i][2], list2[i][2], list3[i][2])]
        list1[i][3] = [(g + h + z) / number for g, h, z in zip(list1[i][3], list2[i][3], list3[i][2])]
        list1[i][6] = (list1[i][6] + list2[i][6] + list3[i][6]) / number
        list1[i][8] = (list1[i][8] + list2[i][8] + list3[i][8]) / number
        list1[i][9] = (list1[i][9] + list2[i][9] + list3[i][9]) / number
        list1[i][10] = (list1[i][10] + list2[i][10] + list3[i][10]) / number
        list1[i][11] = (list1[i][11] + list2[i][11] + list3[i][11]) / number
        list1[i][12] = (list1[i][12] + list2[i][12] + list3[i][12]) / number
        list1[i][13] = (list1[i][13] + list2[i][13] + list3[i][13]) / number

    return list1


def quantitative_analysis(result_list):
    list = []
    for item in result_list:
        print(item)
        map_list = glob(opj(item, "map*"))
        map_list.sort()
        brainmask_NoCSF_list = glob(opj(DATA_PATH, "brainmask_withoutCSF*"))
        brainmask_NoCSF_list.sort()
        tumor_list = glob(opj(DATA_PATH, "mask*"))
        tumor_list.sort()
        f1_list, auc_list, fpr_list, tpr_list, acc_list, threshold_list, f1_opening_list, acc_opening_list = [], [], [], [], [], [], [], []
        name_epochs = item[item.find('max_epochs=') + len('max_epochs='):item.rfind('-latent_dim')]
        name_latentdim = item[item.find('latent_dim=') + len('latent_dim='):item.rfind('.ckpt')]
        if name_latentdim.endswith('-v1'):
            name_latentdim = name_latentdim[:-3]

        for i in range(len(map_list)):
            map = np.load(map_list[i])
            brainmask_NoCSF = np.load(brainmask_NoCSF_list[i])
            tumor = np.load(tumor_list[i])
            f1, auc, fpr, tpr, acc, threshold, f1_opening, acc_opening = roc(map, brainmask_NoCSF, tumor)
            f1_list.append(f1)
            threshold_list.append(threshold)
            acc_list.append(acc)
            auc_list.append(auc)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            f1_opening_list.append(f1_opening)
            acc_opening_list.append(acc_opening)

        f1_average = mean(f1_list)
        f1_opening_average = mean(f1_opening_list)
        auc_average = mean(auc_list)
        acc_average = mean(acc_list)
        acc_opening_average = mean(acc_opening_list)
        threshold_average = mean(threshold_list)
        st_dev = statistics.pstdev(auc_list)
        st_dev_f1 = statistics.pstdev(f1_list)
        st_dev_f1opening = statistics.pstdev(f1_opening_list)
        fpr_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*fpr_list)]
        tpr_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*tpr_list)]

        # If new values are added, check to avarage them in average_results()
        locals()['latent_space' + name_latentdim+'epochs' + name_epochs] = [f1_average, auc_average, fpr_average, tpr_average,
                                                                            name_latentdim, name_epochs, st_dev, name_latentdim+name_epochs,
                                                                            acc_average, threshold_average, acc_opening_average, f1_opening_average, st_dev_f1, st_dev_f1opening]
        list.append(locals()['latent_space' + name_latentdim + 'epochs'+name_epochs])

    return list

# ---------------------------------------------------------------------------------------------------------- #

"""

Add all results to compare
!!! Check, that only the desired runs are chosen by the result_list !!!
---> Check the path and that both paths contain the same datasets

+ Where to save the analysis_results?

"""

analysis_results = "/work/scratch/ecke/Masterarbeit/Results/Baseline_Validation1+2+3_withErosion"

result_list = glob(opj("/work/scratch/ecke/Masterarbeit/Results/Baseline_Validation1", "VanillaVAE*"))
list_run1 = quantitative_analysis(result_list)
list_run1 = sorted(list_run1, key=itemgetter(7))
df_1 = pd.DataFrame(list_run1)
df_1.to_csv(opj(analysis_results, 'raw_data1.csv'), index=False, header=False)

result_list = glob(opj("/work/scratch/ecke/Masterarbeit/Results/Baseline_Validation2", "VanillaVAE*"))
list_run2 = quantitative_analysis(result_list)
list_run2 = sorted(list_run2, key=itemgetter(7))
df_2 = pd.DataFrame(list_run2)
df_2.to_csv(opj(analysis_results, 'raw_data2.csv'), index=False, header=False)

result_list = glob(opj("/work/scratch/ecke/Masterarbeit/Results/Baseline_Validation3", "VanillaVAE*"))
list_run3 = quantitative_analysis(result_list)
list_run3 = sorted(list_run3, key=itemgetter(7))
df_3 = pd.DataFrame(list_run3)
df_3.to_csv(opj(analysis_results, 'raw_data3.csv'), index=False, header=False)

# ---------------------------------------------------------------------------------------------------------- #

# average
average_list = average_results(list_run1, list_run2, list_run3, number = 3)
# sorted from worst to best
list = sorted(average_list, key=itemgetter(1))
print_curve(list[0:1]+list[-1:])
# Save to excel
df = pd.DataFrame(list)
df.to_csv(opj(analysis_results, 'quantitative_analysis_averaged.csv'), index=False, header=False)
