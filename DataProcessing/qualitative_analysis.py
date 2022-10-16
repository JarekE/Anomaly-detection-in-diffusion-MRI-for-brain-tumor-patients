"""

Idea:
Load specific images and print a qualitative output.
Images (good, bad) can be evaluated firstly by looking at normal postprocessing output.

Save images in folder for quantitative analysis of the regarding approach.

!Choose data and name of images for each run separately!

"""
import numpy as np
from scipy.ndimage.morphology import binary_opening
import matplotlib.pyplot as plt
from scipy import ndimage
from os.path import join as opj


# ----------------------------------------------------------------
# Decide!
image_data = "/work/scratch/ecke/Masterarbeit/Results_RecDiscNet/Results_Anomalies/Run4/RecDisc-epoch=244-val_loss=0.12-r=4-an=Mix-d=Half.ckpt"
single = False
RecDiscNet = True
analysis_results = "/work/scratch/ecke/Masterarbeit/"
input_list, mask_list, output_list, brainmask_edited_list, brainmask_list, output_03_list, output_MO_list = [], [], [], [], [], [], []

# Get this dataset
dataset = ['1', '6']
slide = [16, 40, 13, 32]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ----------------------------------------------------------------
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
    fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(10, 5))
    image = [input_list, mask_list, output_list]
    #image_name = ['Data', 'Mask', 'Map']
    #tumor = [1, 6]
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


