# Load names of results and use them as excel sheet

import pandas as pd
from glob import glob
from os.path import join as opj
import numpy as np
from skimage import filters
from sklearn.metrics import roc_curve, f1_score, roc_auc_score
import statistics
from statistics import mean
import matplotlib.pyplot as plt

# Results to Excel
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

def logs_for_training(mode = "RecDisc"):

    if mode == "DAE":
        load_logs = "/work/scratch/ecke/Masterarbeit/Results_DAE/Logs/LogsForTrainingVanVAE"
        load_list = glob(opj(load_logs, "*"))
        color_list = ['b', 'g', 'y', 'r', 'c']
        epochs_number = range(1, 501)
        loss = np.zeros(500)

        for i, excel in enumerate(load_list):
            epochs = pd.read_csv(opj(load_logs, excel)).val_loss.dropna().to_numpy()
            plt.plot(epochs_number, epochs, color_list[i], label=i)
            print(epochs.shape)
            loss = np.add(loss, epochs)

        plt.title('Check VanVAE loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        mean_loss = loss / len(load_list)

        load_logs = "/work/scratch/ecke/Masterarbeit/Results_DAE/Logs/LogsForTrainingUNET"
        load_list = glob(opj(load_logs, "*"))
        loss = np.zeros(500)

        for i, excel in enumerate(load_list):
            epochs = pd.read_csv(opj(load_logs, excel)).val_loss.dropna().to_numpy()
            plt.plot(epochs_number, epochs, color_list[i], label=i)
            print(epochs.shape)
            loss = np.add(loss, epochs)

        plt.title('Check U-Net loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        mean_loss_2 = loss / len(load_list)
        plt.rcParams["figure.figsize"] = (10, 4)
        plt.plot(epochs_number, mean_loss, 'navy', label='VanVAE Loss')
        plt.plot(epochs_number, mean_loss_2, 'royalblue', label='U-Net Loss')
        plt.title('Mean Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('DAE_loss.pdf')
        plt.show()

        mean_loss = mean_loss[0:50]
        mean_loss_2 = mean_loss_2[0:50]
        epochs_number = range(1, 51)
        plt.rcParams["figure.figsize"] = (10, 4)
        plt.plot(epochs_number, mean_loss, 'navy', label='VanVAE Loss')
        plt.plot(epochs_number, mean_loss_2, 'royalblue', label='U-Net Loss')
        plt.title('Mean Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('DAE_loss_zoomed.pdf')
        plt.show()
    else:
        load_logs = "/work/scratch/ecke/Masterarbeit/Results_RecDiscNet/LogsForTraining"
        load_list = glob(opj(load_logs, "*"))
        loss = np.zeros(250)
        loss_1 = np.zeros(250)
        loss_2 = np.zeros(250)
        epochs_number = range(1, 251)

        for i, excel in enumerate(load_list):
            val_loss = pd.read_csv(opj(load_logs, excel)).val_loss.dropna().to_numpy()
            val_DL_loss = pd.read_csv(opj(load_logs, excel)).val_DL.dropna().to_numpy()
            val_RL_loss = pd.read_csv(opj(load_logs, excel)).val_RL.dropna().to_numpy()

            #plt.plot(epochs_number, val_loss, 'b', label="Loss "+str(i))
            #plt.plot(epochs_number, val_DL_loss, 'r', label="DL Loss "+str(i))
            #plt.plot(epochs_number, val_RL_loss,'y', label="RL Loss "+str(i))

            print(val_loss.shape, val_DL_loss.shape, val_RL_loss.shape)
            loss = np.add(loss, val_loss)
            DL_loss = np.add(loss_1, val_DL_loss)
            RL_loss = np.add(loss_2, val_RL_loss)

        if 0:
            plt.title('Check U-Net loss')
            plt.ylim([0, 0.2])
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        mean_loss = loss / len(load_list)
        mean_loss_DL = DL_loss / len(load_list)
        mean_loss_RL = RL_loss / len(load_list)

        #plt.style.use('seaborn-whitegrid')
        plt.rcParams["figure.figsize"] = (10, 4)
        plt.plot(epochs_number, mean_loss, 'navy', label='Loss')
        plt.plot(epochs_number, mean_loss_DL, 'royalblue', label='Discrimination Loss')
        plt.plot(epochs_number, mean_loss_RL, 'cornflowerblue', label='Reconstruction Loss')
        plt.title('Mean Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('RecDiscNet_loss.pdf')
        plt.show()

        mean_loss = mean_loss[0:50]
        mean_loss_DL = mean_loss_DL[0:50]
        mean_loss_RL = mean_loss_RL[0:50]
        epochs_number = range(1, 51)

        plt.rcParams["figure.figsize"] = (10, 4)
        plt.plot(epochs_number, mean_loss, 'navy', label='Loss')
        plt.plot(epochs_number, mean_loss_DL, 'royalblue', label='Discrimination Loss')
        plt.plot(epochs_number, mean_loss_RL, 'cornflowerblue', label='Reconstruction Loss')
        plt.title('Mean Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('RecDiscNet_loss_zoomed.pdf')
        plt.show()

    return

logs_for_training()
