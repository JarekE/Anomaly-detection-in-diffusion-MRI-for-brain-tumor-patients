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
# Get this run
image_data = "/work/scratch/ecke/Masterarbeit/Results/Baseline_Validation1/VanillaVAE-epoch=84-val_loss=0.02-max_epochs=200-latent_dim=64.ckpt"

# Get this dataset
dataset = '7'

# Save qualitative images here
analysis_results = "/work/scratch/ecke/Masterarbeit/Results/Baseline_Validation1+2+3_withErosion"
name= "64_200_problemTumor"
# ----------------------------------------------------------------

input = np.load("/work/scratch/ecke/Masterarbeit/Data/Test/vp"+dataset+".npy")
input = np.mean(input, axis=0)
input = np.flip(np.flip(np.swapaxes(input, 0, 2), axis=0), axis=1)
mask = np.load("/work/scratch/ecke/Masterarbeit/Data/Test/maskvp"+dataset+".npy")
mask = np.flip(np.flip(np.swapaxes(mask, 0, 2), axis=0), axis=1)
mask[mask == 0] = np.nan
output = np.load(image_data+"/map_vp"+dataset+".npy")
output = np.flip(np.flip(np.swapaxes(output, 0, 2), axis=0), axis=1)
# Only for denoising autoencoder (other datasamples are already pre-processed with mask):
# DELETE LATER
if 0:
    brainmask_edited = np.load("/work/scratch/ecke/Masterarbeit/Data/Test/brainmask_withoutCSFvp"+dataset+".npy")
    brainmask_edited = np.flip(np.flip(np.swapaxes(brainmask_edited, 0, 2), axis=0), axis=1)
    brainmask = np.load("/work/scratch/ecke/Masterarbeit/Data/Test/b0_brainmaskvp"+dataset+".npy")
    brainmask = np.flip(np.flip(np.swapaxes(brainmask, 0, 2), axis=0), axis=1)

    brainmask = ndimage.binary_erosion(brainmask, structure=np.ones((2,2,2))).astype(brainmask.dtype)

    input = np.where(brainmask > 0, input, 0)
    mask = np.where(brainmask > 0, mask, 0)
    mask[mask == 0] = np.nan
    output = np.where(brainmask_edited == 1, output, 0)

output_05 = np.where(output >= 0.5, 1, 0)
output_MO = binary_opening(output_05, structure=np.ones((3, 3, 3)))



fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 4
# Define slides of image to print
slide_healthy = 16
slide_tumor = 25


fig.add_subplot(rows, columns, 1)
plt.imshow(input[slide_healthy, :, :], cmap='gray')
plt.axis('off')
plt.title("Data", fontsize=25)

fig.add_subplot(rows, columns, 2)
plt.imshow(input[slide_healthy, :, :], cmap='gray')
plt.imshow(mask[slide_healthy, :, :], vmin=0, vmax=4, alpha=1)
plt.axis('off')
plt.title("Ground Truth", fontsize=25)

fig.add_subplot(rows, columns, 3)
plt.imshow(output_05[slide_healthy, :, :], cmap='gray')
plt.axis('off')
plt.title("Binary", fontsize=25)

fig.add_subplot(rows, columns, 4)
plt.imshow(output_MO[slide_healthy, :, :], cmap='gray')
plt.axis('off')
plt.title("Opening", fontsize=25)

fig.add_subplot(rows, columns, 5)
plt.imshow(input[slide_tumor, :, :], cmap='gray')
plt.axis('off')

fig.add_subplot(rows, columns, 6)
plt.imshow(input[slide_tumor, :, :], cmap='gray')
plt.imshow(mask[slide_tumor, :, :], vmin=0, vmax=4, alpha=1)
plt.axis('off')

fig.add_subplot(rows, columns, 7)
plt.imshow(output_05[slide_tumor, :, :], cmap='gray')
plt.axis('off')

fig.add_subplot(rows, columns, 8)
plt.imshow(output_MO[slide_tumor, :, :], cmap='gray')
plt.axis('off')

plt.show()
fig.savefig(opj(analysis_results, name+"tumor"+dataset+"slide_healthy"+str(slide_healthy)+"slide_tumor"+str(slide_tumor)))










