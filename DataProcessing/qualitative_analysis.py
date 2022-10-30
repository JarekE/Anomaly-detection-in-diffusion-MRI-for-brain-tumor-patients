import numpy as np
from scipy.ndimage.morphology import binary_opening
import matplotlib.pyplot as plt
from scipy import ndimage
import skimage.measure
from sklearn.metrics import roc_curve,  precision_recall_curve
from skimage import filters

#####################################
#
#
# Plot qualitative results for the final output data of the models
#
#
#####################################

# Used for RecDiscNet output
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Print array of images as used in thesis (decide for path and network)
def best_results_images():
    image_data = ["/work/scratch/ecke/Masterarbeit/Results_DAE/Results/Run4/VanillaVAE-epoch=373-val_loss=0.02-r=4-ldim=128-ar=Sigmoid.ckpt",
                  "/work/scratch/ecke/Masterarbeit/Results_DAE/Results/Run4/UNet-epoch=222-val_loss=0.16-r=4-ar=Sigmoid-f=16.ckpt"]
    RecDiscNet = False
    analysis_results = "/work/scratch/ecke/Masterarbeit/"
    input_list, mask_list, output_list, brainmask_edited_list, brainmask_list, output_threshold_list, output_MO_list, threshold_list = \
        [[] for _ in range(len(image_data))], [[] for _ in range(len(image_data))], [[] for _ in range(len(image_data))], \
        [[] for _ in range(len(image_data))], [[] for _ in range(len(image_data))], [[] for _ in range(len(image_data))], \
        [[] for _ in range(len(image_data))], [[] for _ in range(len(image_data))]

    # Get this dataset
    dataset = ['31', '2', '30']
    # Get this slide
    slide = [24, 22, 28]
    best_thresholds = ['Dice', 'Youden']
    name_of_image = 'example_VanVAE7_dice_UNet5_youdens_supervised.png'

    for s in range(len(image_data)):
        for i in range(len(dataset)):
            input_list[s].append(np.flip(np.flip(
                np.swapaxes(np.mean(np.load("/work/scratch/ecke/Masterarbeit/Data/Test/vp" + dataset[i] + ".npy"), axis=0),
                            0, 2), axis=0), axis=1))
            mask_list[s].append(np.flip(np.flip(
                np.swapaxes(np.load("/work/scratch/ecke/Masterarbeit/Data/Test/maskvp" + dataset[i] + ".npy"), 0, 2),
                axis=0), axis=1))
            output_list[s].append(
                np.flip(np.flip(np.swapaxes(np.load(image_data[s] + "/map_vp" + dataset[i] + ".npy"), 0, 2), axis=0), axis=1))
            if RecDiscNet == True:
                output_list[s][i] = sigmoid(output_list[s][i])

            brainmask_edited_list[s].append(np.flip(np.flip(np.swapaxes(
                np.load("/work/scratch/ecke/Masterarbeit/Data/Test/brainmask_withoutCSFvp" + dataset[i] + ".npy"), 0,
                2), axis=0), axis=1))
            brainmask_list[s].append(np.flip(np.flip(
                np.swapaxes(np.load("/work/scratch/ecke/Masterarbeit/Data/Test/b0_brainmaskvp" + dataset[i] + ".npy"),
                            0, 2), axis=0), axis=1))

            brainmask_list[s][i] = ndimage.binary_erosion(brainmask_list[s][i], structure=np.ones((2, 2, 2))).astype(
                brainmask_list[s][i].dtype)

            input_list[s][i] = np.where(brainmask_list[s][i] > 0, input_list[s][i], 0)
            mask_list[s][i] = np.where(brainmask_list[s][i] > 0, mask_list[s][i], 0)
            output_list[s][i] = np.where(brainmask_edited_list[s][i] == 1, output_list[s][i], 0)

            if i == 0:
                for threshold in best_thresholds:
                    if threshold == 'Dice':
                        flatten_opf1 = output_list[s][i].flatten()
                        tumormask_flatten_opf1 = mask_list[s][i].flatten()
                        tumormask_flatten_opf1 = tumormask_flatten_opf1[flatten_opf1 != 0]
                        tumormask_flatten_opf1[tumormask_flatten_opf1 > 1] = 1
                        flatten_opf1 = flatten_opf1[flatten_opf1 != 0]
                        precision, recall, thresholds = precision_recall_curve(tumormask_flatten_opf1, flatten_opf1)
                        optimal_threshold = thresholds[np.argmax(np.nan_to_num((2 * precision * recall) / (precision + recall)))]
                        threshold_list[s].append((optimal_threshold))
                    elif threshold == 'Youden':
                        flatten_opauc = output_list[s][i].flatten()
                        tumormask_flatten_opauc = mask_list[s][i].flatten()
                        tumormask_flatten_opauc = tumormask_flatten_opauc[flatten_opauc != 0]
                        tumormask_flatten_opauc[tumormask_flatten_opauc > 1] = 1
                        flatten_opauc = flatten_opauc[flatten_opauc != 0]
                        fpr, tpr, thresholds = roc_curve(tumormask_flatten_opauc, flatten_opauc)
                        optimal_threshold = thresholds[np.argmax(tpr - fpr)]
                        threshold_list[s].append((optimal_threshold))
                    elif threshold == 'Mean Pooling':
                        pool_map = skimage.measure.block_reduce(output_list[s][i], (16, 16, 16), np.mean)
                        optimal_threshold = np.max(pool_map)
                        threshold_list[s].append((optimal_threshold))
                    elif threshold == "Otsu":
                        flatten_otsu = output_list[s][i].flatten()
                        optimal_threshold = filters.threshold_otsu(flatten_otsu[flatten_otsu != 0])
                        threshold_list[s].append((optimal_threshold))
                    else:
                        print("Not implemented yet")

            output_threshold_list[s].append(np.where(output_list[s][i] >= threshold_list[s][0], 1, 0))
            output_MO_list[s].append(binary_opening(np.where(output_list[s][i] >= threshold_list[s][1], 1, 0), structure=np.ones((3, 3, 3))))
            mask_list[s][i][mask_list[s][i] == 0] = np.nan

    image = [input_list, output_list, output_threshold_list, output_MO_list]

    if len(image_data) == 1:
        fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(6.6, 6))
        for s, number in enumerate(slide):
            for i, img in enumerate(image):
                if i == 0:
                    ax[s, i].imshow(img[0][s][number, :, :], cmap='gray', vmin=0, vmax=1)
                    ax[s, i].imshow(mask_list[0][s][number, :, :], vmin=0, vmax=4, alpha=1)
                    ax[s, i].axis('off')
                elif i == 1:
                    ax[s, i].imshow(img[0][s][number, :, :], cmap='inferno', vmin=0, vmax=1)
                    ax[s, i].axis('off')
                elif i == 2:
                    ax[s, i].imshow(img[0][s][number, :, :], cmap='gray', vmin=0, vmax=1)
                    ax[s, i].axis('off')
                else:
                    ax[s, i].imshow(img[0][s][number, :, :], cmap='gray', vmin=0, vmax=1)
                    ax[s, i].axis('off')
    else:
        fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(8, 6))
        for s, number in enumerate(slide):
            ax[s, 0].imshow(input_list[0][s][number, :, :], cmap='gray', vmin=0, vmax=1)
            ax[s, 0].imshow(mask_list[0][s][number, :, :], vmin=0, vmax=4, alpha=1)
            ax[s, 0].axis('off')
            ax[s, 1].imshow(output_list[0][s][number, :, :], cmap='inferno', vmin=0, vmax=1)
            ax[s, 1].axis('off')
            ax[s, 2].imshow(output_threshold_list[0][s][number, :, :], cmap='gray', vmin=0, vmax=1)
            ax[s, 2].axis('off')
            ax[s, 3].imshow(output_list[1][s][number, :, :], cmap='inferno', vmin=0, vmax=1)
            ax[s, 3].axis('off')
            ax[s, 4].imshow(output_MO_list[1][s][number, :, :], cmap='gray', vmin=0, vmax=1)
            ax[s, 4].axis('off')

    plt.tight_layout()
    plt.savefig(analysis_results + name_of_image, transparent=True)
    plt.show()
    plt.close(fig)
    return

# Single image of choice (decide path, dataset and slide)
def qualitative_images():
    image_data = "/work/scratch/ecke/Masterarbeit/Results_RecDiscNet/Results_Anomalies/Run4/RecDisc-epoch=244-val_loss=0.12-r=4-an=Mix-d=Half.ckpt"
    single = False
    RecDiscNet = True
    analysis_results = "/work/scratch/ecke/Masterarbeit/"
    input_list, mask_list, output_list, brainmask_edited_list, brainmask_list, output_03_list, output_MO_list = [], [], [], [], [], [], []

    dataset = ['1', '6']
    slide = [16, 40, 13, 32]

    for i in range(len(dataset)):
        input_list.append(np.flip(np.flip(np.swapaxes(np.mean(np.load("/work/scratch/ecke/Masterarbeit/Data/Test/vp"+dataset[i]+".npy"), axis=0), 0, 2), axis=0), axis=1))
        mask_list.append(np.flip(np.flip(np.swapaxes(np.load("/work/scratch/ecke/Masterarbeit/Data/Test/maskvp"+dataset[i]+".npy"), 0, 2), axis=0), axis=1))
        output_list.append(np.flip(np.flip(np.swapaxes(np.load(image_data+"/map_vp"+dataset[i]+".npy"), 0, 2), axis=0), axis=1))
        if RecDiscNet == True:
            output_list[i] = sigmoid(output_list[i])

        if 1:
            brainmask_edited_list.append(np.flip(np.flip(np.swapaxes(np.load("/work/scratch/ecke/Masterarbeit/Data/Test/brainmask_withoutCSFvp"+dataset[i]+".npy"), 0, 2), axis=0), axis=1))
            brainmask_list.append(np.flip(np.flip(np.swapaxes(np.load("/work/scratch/ecke/Masterarbeit/Data/Test/b0_brainmaskvp"+dataset[i]+".npy"), 0, 2), axis=0), axis=1))

            brainmask_list[i] = ndimage.binary_erosion(brainmask_list[i], structure=np.ones((2,2,2))).astype(brainmask_list[i].dtype)

            input_list[i] = np.where(brainmask_list[i] > 0, input_list[i], 0)
            mask_list[i] = np.where(brainmask_list[i] > 0, mask_list[i], 0)
            mask_list[i][mask_list[i] == 0] = np.nan
            output_list[i] = np.where(brainmask_edited_list[i] == 1, output_list[i], 0)

        output_03_list.append(np.where(output_list[i] >= 0.3, 1, 0))
        output_MO_list.append(binary_opening(output_03_list[i], structure=np.ones((3, 3, 3))))

    if single == True:
        slide_healthy = 16
        slide_tumor = 40

        image = [input_list[0], output_list[0], output_03_list[0], output_MO_list[0], mask_list[0]]
        image_name = ['Data', 'Map', 'Binary', 'Opening', 'Mask']
        slide = [slide_healthy, slide_tumor]
        slide_name = ['Healthy', 'Tumor']

        for i, img in enumerate(image):
            for s, data in enumerate(slide):
                if image_name[i] == 'Data':
                    plt.imshow(img[data, :, :], cmap='gray', vmin=0, vmax=1)
                    plt.axis('off')
                    plt.savefig(analysis_results+str(slide_name[s])+str(data)+'_Data.png', transparent=True)
                    plt.close()
                elif image_name[i] == 'Map':
                    plt.imshow(img[data, :, :], cmap='inferno', vmin=0, vmax=1)
                    plt.axis('off')
                    plt.savefig(analysis_results+str(slide_name[s]) + str(data) + '_Map.png', transparent=True)
                    plt.close()
                elif image_name[i] == 'Binary':
                    plt.imshow(img[data, :, :], cmap='gray')
                    plt.axis('off')
                    plt.savefig(analysis_results+str(slide_name[s]) + str(data) + '_Binary.png', transparent=True)
                    plt.close()
                elif image_name[i] == 'Opening':
                    plt.imshow(img[data, :, :], cmap='gray')
                    plt.axis('off')
                    plt.savefig(analysis_results+str(slide_name[s]) + str(data) + '_Opening.png', transparent=True)
                    plt.close()
                elif image_name[i] == 'Mask':
                    plt.imshow(img[data, :, :], vmin=0, vmax=4, alpha=1)
                    plt.axis('off')
                    plt.savefig(analysis_results+str(slide_name[s]) + str(data) + '_GT.png', transparent=True)
                    plt.close()
    else:
        image = [input_list, mask_list, output_list]
        fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(10, 4.2))

        for d in range(len(dataset)):
            if d == 0:
                for s, number in enumerate(slide[0:2]):
                    for i, img in enumerate(image):
                        if i == 1:
                            ax[s, i].imshow(np.zeros((80,64)), cmap='gray')
                            ax[s, i].imshow(img[d][number, :, :], vmin=0, vmax=4, alpha=1)
                            ax[s, i].axis('off')
                        elif i == 2:
                            ax[s, i].imshow(img[d][number, :, :], cmap='inferno', vmin=0, vmax=1)
                            ax[s, i].axis('off')
                        else:
                            ax[s, i].imshow(img[d][number, :, :], cmap='gray', vmin=0, vmax=1)
                            ax[s, i].axis('off')
            if d == 1:
                for s, number in enumerate(slide[2:4]):
                    for i, img in enumerate(image):
                        if i == 1:
                            ax[s, i+3].imshow(np.zeros((80, 64)), cmap='gray')
                            ax[s, i+3].imshow(img[d][number, :, :], vmin=0, vmax=4, alpha=1)
                            ax[s, i+3].axis('off')
                        elif i == 2:
                            ax[s, i+3].imshow(img[d][number, :, :], cmap='inferno', vmin=0, vmax=1)
                            ax[s, i+3].axis('off')
                        else:
                            ax[s, i+3].imshow(img[d][number, :, :], cmap='gray', vmin=0, vmax=1)
                            ax[s, i+3].axis('off')
        plt.tight_layout()
        plt.savefig(analysis_results+'example.png', transparent=True)
        plt.show()
        plt.close(fig)
    return