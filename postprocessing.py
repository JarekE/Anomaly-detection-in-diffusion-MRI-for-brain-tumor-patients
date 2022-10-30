import os
import shutil
from glob import glob
from os.path import join as opj
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import montage
from scipy import ndimage
from scipy.ndimage.morphology import binary_opening
import config


def postprocessing_baseline(result, input, brainmask, name, mask):
    result /= result.max()
    map = np.subtract(input, result)
    map_mean = np.mean(map, axis=0)
    map_mean = np.absolute(map_mean)
    map_mean[brainmask == 0] = 0

    result_images(map_mean, name, mask, cmap="hot", note="hot")
    result_images(map_mean, name, mask, cmap="gray", note="gray")
    result_images(binary_opening(np.where(map_mean >= 0.5, 1, 0), structure=np.ones((3,3,3))).astype(int), name, mask, cmap="gray", note="opening")

    print("Save map: ", name)
    np.save(opj(config.results_path, 'map_') + name, map_mean)

def postprocessing_reddisc(result, input, brainmask, name, mask):
    map = result[0,...]
    result_images(map, name, mask, cmap="hot", note="hot")
    result_images(map, name, mask, cmap="gray", note="gray")
    result_images(np.where(map>0.5, 1, 0), name, mask, cmap="gray", note="BW")

    print("Save map: ", name)
    np.save(opj(config.results_path, 'map_') + name, map)

def prepare_results():
    path = config.save_path
    if os.path.exists(config.results_path):
        shutil.rmtree(config.results_path)
    os.mkdir(config.results_path)
    return path

def result_images(map, name, mask, cmap, note):
    map = np.flip(np.flip(np.swapaxes(map, 0, 2), axis=0), axis=1)

    if cmap == "hot":
        mask_dil = ndimage.binary_dilation(mask, iterations=2)
        mask_contour = mask_dil - mask

        # Rotate image (not data!) for optimal visual output
        mask_contour = np.flip(np.flip(np.swapaxes(mask_contour, 0, 2), axis=0), axis=1)

        fig, ax1 = plt.subplots(1, 1, figsize=(20, 20))
        ax1.imshow(montage(map), cmap=cmap)
        mask_contour = mask_contour.astype(float)
        mask_contour[mask_contour > 0.5] = 1
        mask_contour[mask_contour <= 0.5] = np.nan
        ax1.imshow(montage(mask_contour), 'binary', alpha=1)
        path = opj(config.results_path, 'image_map_' + name + note + '.png')
        fig.savefig(path)
        plt.close(fig)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(20, 20))
        ax1.imshow(montage(map), cmap=cmap)
        path = opj(config.results_path, 'image_map_' + name + note + '.png')
        fig.savefig(path)
        plt.close(fig)
    return

def processing():
    input_path = config.test
    mask_path = config.test_mask
    brainmask_path = glob(opj('/work/scratch/ecke/Masterarbeit/Data/Test', 'brainmask_withoutCSF*'))
    brainmask_path.sort()
    output_path = glob(opj(config.results_path, "output_*"))
    output_path.sort()

    for i in range(len(input_path)):
        input = np.load(input_path[i])
        mask = np.load(mask_path[i])
        brainmask = np.load(brainmask_path[i])
        output = np.load(output_path[i])
        name = input_path[i].split("/")[-1]
        print(name)

        if config.network == "VanillaVAE" or config.network == "UNet":
            postprocessing_baseline(output, input, brainmask, name, mask)
        if config.network == "RecDisc":
            postprocessing_reddisc(output, input, brainmask, name, mask)
    return