from statistics import mean
from glob import glob
from os.path import join as opj
import numpy as np
import matplotlib.pyplot as plt
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
        print('Model' + str(list[i][7]) + ': ROC AUC=%.3f' % (list[i][1]))
        plt.plot(list[i][2], list[i][3], marker=".",
                    label='Model' + str(list[i][7]) + ' AUC: ' + str(np.around(list[i][1], 4)) + '\u00B1' + str(round(list[i][6], 3)))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(opj(analysis_results, 'curve.png'), bbox_inches='tight', transparent = True)
    plt.show()


    # zoomed
    for i in range(len(list)):
        print('Model' + str(list[i][7]) + ': ROC AUC=%.3f' % (list[i][1]))
        plt.plot(list[i][2], list[i][3], marker=".",
                    label='Model' + str(list[i][7]) + ' AUC: ' + str(np.around(list[i][1], 4)) + ' \u00B1 ' + str(round(list[i][6], 3)))

    plt.xlabel('False Positive Rate (zoomed in)')
    plt.ylabel('True Positive Rate (zoomed in)')
    plt.axis([0.1, 0.5, 0.3, 0.8])
    plt.legend()
    plt.savefig(opj(analysis_results, 'curve_zoomed.png'), bbox_inches='tight', transparent = True)
    plt.show()


def average_results(list1, list2, list3, list4, list5, list6, list7, number):

    for i in range(len(list1)):
        list1[i][0] = (list1[i][0] + list2[i][0] + list3[i][0] + list4[i][0] + list5[i][0] + list6[i][0] + list7[i][0]) / number
        list1[i][1] = (list1[i][1] + list2[i][1] + list3[i][1] + list4[i][1] + list5[i][1] + list6[i][1] + list7[i][1]) / number
        list1[i][2] = [(g + h + z + a + b + c + d) / number for g, h, z, a, b, c, d in zip(list1[i][2], list2[i][2], list3[i][2], list4[i][2], list5[i][2], list6[i][2], list7[i][2])]
        list1[i][3] = [(g + h + z + a + b + c + d) / number for g, h, z, a, b, c, d in zip(list1[i][3], list2[i][3], list3[i][3], list4[i][3], list5[i][3], list6[i][3], list7[i][3])]
        list1[i][6] = (list1[i][6] + list2[i][6] + list3[i][6] + list4[i][6] + list5[i][6] + list6[i][6] + list7[i][6]) / number
        list1[i][8] = (list1[i][8] + list2[i][8] + list3[i][8] + list4[i][8] + list5[i][8] + list6[i][8] + list7[i][8]) / number
        list1[i][9] = (list1[i][9] + list2[i][9] + list3[i][9] + list4[i][9] + list5[i][9] + list6[i][9] + list7[i][9]) / number
        list1[i][10] = (list1[i][10] + list2[i][10] + list3[i][10] + list4[i][10] + list5[i][10] + list6[i][10] + list7[i][10]) / number
        list1[i][11] = (list1[i][11] + list2[i][11] + list3[i][11] + list4[i][11] + list5[i][11] + list6[i][11] + list7[i][11]) / number
        list1[i][12] = (list1[i][12] + list2[i][12] + list3[i][12] + list4[i][12] + list5[i][12] + list6[i][12] + list7[i][12]) / number
        list1[i][13] = (list1[i][13] + list2[i][13] + list3[i][13] + list4[i][13] + list5[i][13] + list6[i][13] + list7[i][13]) / number

    return list1

def normalize(data):
    sorted = np.sort(data, axis=None)
    index = int(len(sorted)/1000)
    data = np.where(data > sorted[-index], sorted[-index], data)
    data = data/data.max()
    return data

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
        network = item.split("/")[-1].split("-")[0]
        f1_list, auc_list, fpr_list, tpr_list, acc_list, threshold_list, f1_opening_list, acc_opening_list, histo_list = [], [], [], [], [], [], [], [], []
        # Naming -> depends on Network and paramter
        if network == "VanillaVAE":
            para1 = item[item.find('ldim=') + len('ldim='):item.rfind('-ar')]
            para2 = item[item.find('ar=') + len('ar='):item.rfind('.ckpt')]
            if para2.endswith('-v1'):
                para2 = para2[:-3]
            para3 = network
        elif network == "UNet":
            para1 = item[item.find('ar=') + len('ar='):item.rfind('-f')]
            para2 = item[item.find('f=') + len('f='):item.rfind('.ckpt')]
            if para2.endswith('-v1'):
                para2 = para2[:-3]
            para3 = network
        # RedDisc and RedDiscUnet
        else:
            para1 = item[item.find('XXX=') + len('XXX='):item.rfind('-XXX')]
            para2 = item[item.find('XXX=') + len('XXX='):item.rfind('-XXX')]
            para3 = item[item.find('XXX=') + len('XXX='):item.rfind('.ckpt')]
            if para3.endswith('-v1'):
                para3 = para3[:-3]

        for i in range(len(map_list)):
            map = np.load(map_list[i])
            if network == "VanillaVAE" and para2 == "None":
                map = normalize(map)
            histo_list.append(map)
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

        # Show Histogram
        histo_data = np.vstack(histo_list).flatten()
        all_elements = len(histo_data)
        histo_data = histo_data[histo_data > 0.1]
        percentage_elements = len(histo_data)/all_elements*100
        plt.hist(histo_data, label='Model_'+para3+'_perc_'+str(int(percentage_elements)), bins = 30, density=True)
        plt.legend()
        plt.show()

        # Averaging
        f1_average = mean(f1_list)
        f1_opening_average = mean(f1_opening_list)
        auc_average = mean(auc_list)
        acc_average = mean(acc_list)
        acc_opening_average = mean(acc_opening_list)
        threshold_average = mean(threshold_list)
        fpr_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*fpr_list)]
        tpr_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*tpr_list)]

        # Standard deviation
        st_dev = statistics.pstdev(auc_list)
        st_dev_f1 = statistics.pstdev(f1_list)
        st_dev_f1opening = statistics.pstdev(f1_opening_list)

        # Save information for this image
        list.append([f1_average, auc_average, fpr_average, tpr_average, para1, para2, st_dev,
                     para3+'_'+para1+'_'+para2, acc_average, threshold_average, acc_opening_average,
                     f1_opening_average, st_dev_f1, st_dev_f1opening])

    return list

# ---------------------------------------------------------------------------------------------------------- #

"""

Add all results to compare
!!! Check, that only the desired runs are chosen by the result_list !!!
---> Check the path and that both paths contain the same datasets

+ Where to save the analysis_results?

"""

analysis_results = "/work/scratch/ecke/Masterarbeit/Results_DAE/Results"

result_list = glob(opj("/work/scratch/ecke/Masterarbeit/Results_DAE/Results/Run1", "*"))
list_run1 = quantitative_analysis(result_list)
list_run1 = sorted(list_run1, key=itemgetter(7))
df_1 = pd.DataFrame(list_run1)
df_1.to_csv(opj(analysis_results, 'raw_data1.csv'), index=False, header=False)

result_list = glob(opj("/work/scratch/ecke/Masterarbeit/Results_DAE/Results/Run2", "*"))
list_run2 = quantitative_analysis(result_list)
list_run2 = sorted(list_run2, key=itemgetter(7))
df_2 = pd.DataFrame(list_run2)
df_2.to_csv(opj(analysis_results, 'raw_data2.csv'), index=False, header=False)

result_list = glob(opj("/work/scratch/ecke/Masterarbeit/Results_DAE/Results/Run3", "*"))
list_run3 = quantitative_analysis(result_list)
list_run3 = sorted(list_run3, key=itemgetter(7))
df_3 = pd.DataFrame(list_run3)
df_3.to_csv(opj(analysis_results, 'raw_data3.csv'), index=False, header=False)

result_list = glob(opj("/work/scratch/ecke/Masterarbeit/Results_DAE/Results/Run4", "*"))
list_run4 = quantitative_analysis(result_list)
list_run4 = sorted(list_run4, key=itemgetter(7))
df_4 = pd.DataFrame(list_run4)
df_4.to_csv(opj(analysis_results, 'raw_data4.csv'), index=False, header=False)

result_list = glob(opj("/work/scratch/ecke/Masterarbeit/Results_DAE/Results/Run5", "*"))
list_run5 = quantitative_analysis(result_list)
list_run5 = sorted(list_run5, key=itemgetter(7))
df_5 = pd.DataFrame(list_run5)
df_5.to_csv(opj(analysis_results, 'raw_data5.csv'), index=False, header=False)

result_list = glob(opj("/work/scratch/ecke/Masterarbeit/Results_DAE/Results/Run6", "*"))
list_run6 = quantitative_analysis(result_list)
list_run6 = sorted(list_run6, key=itemgetter(7))
df_6 = pd.DataFrame(list_run6)
df_6.to_csv(opj(analysis_results, 'raw_data6.csv'), index=False, header=False)

result_list = glob(opj("/work/scratch/ecke/Masterarbeit/Results_DAE/Results/Run7", "*"))
list_run7 = quantitative_analysis(result_list)
list_run7 = sorted(list_run7, key=itemgetter(7))
df_7 = pd.DataFrame(list_run7)
df_7.to_csv(opj(analysis_results, 'raw_data7.csv'), index=False, header=False)

# ---------------------------------------------------------------------------------------------------------- #

# average
average_list = average_results(list_run1, list_run2, list_run3, list_run4, list_run5, list_run6, list_run7, number = 7)
# sorted from worst to best
list = sorted(average_list, key=itemgetter(1))
print_curve(list[0:1]+list[-1:])
# Save to excel
df = pd.DataFrame(list)
df.to_csv(opj(analysis_results, 'quantitative_analysis_averaged.csv'), index=False, header=False)