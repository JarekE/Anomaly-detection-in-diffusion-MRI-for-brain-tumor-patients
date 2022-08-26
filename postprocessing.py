# Everything about post-processing
import os
import shutil
from glob import glob
from os.path import join as opj
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import montage
from scipy import ndimage
import pickle
#from skimage.morphology import binary_opening
from scipy.ndimage.morphology import binary_opening

import config


def postprocessing_baseline(result, input, brainmask, name, mask):
    result /= result.max()
    # L1 loss
    map = np.subtract(input, result)
    map_mean = np.mean(map, axis=0)
    map_mean = np.absolute(map_mean)
    # Brainmask !!! without CSF and with erosion !!!
    map_mean[brainmask == 0] = 0

    # Set to 0 for better visualization
    #map_mean[map_mean < 0.2] = 0

    # !!! NO CSF !!!
    result_images(map_mean, name, mask, cmap="hot", note="hot")
    result_images(map_mean, name, mask, cmap="gray", note="gray")
    result_images(binary_opening(np.where(map_mean >= 0.5, 1, 0), structure=np.ones((3,3,3))).astype(int), name, mask, cmap="gray", note="opening")

    print("Save map: ", name)
    np.save(opj(config.results_path, 'map_') + name, map_mean)

def postprocessing_voxel(result, input, brainmask, name, mask):
    result /= result.max()
    # L1 loss
    map = np.subtract(input, result)
    map_mean = np.mean(map, axis=0)
    map_mean = np.absolute(map_mean)
    # Brainmask !!! without CSF and with erosion !!!
    map_mean[brainmask == 0] = 0

    # Set to 0 for better visualization
    #map_mean[map_mean < 0.2] = 0

    # !!! NO CSF !!!
    result_images(map_mean, name, mask, cmap="hot", note="hot")
    result_images(map_mean, name, mask, cmap="gray", note="gray")
    result_images(binary_opening(np.where(map_mean >= 0.5, 1, 0), structure=np.ones((3,3,3))).astype(int), name, mask, cmap="gray", note="opening")

    print("Save map: ", name)
    np.save(opj(config.results_path, 'map_') + name, map_mean)

def postprocessing_reddisc(result, input, brainmask, name, mask):
    result /= result.max()
    # L1 loss
    map = np.subtract(input, result)
    map_mean = np.mean(map, axis=0)
    map_mean = np.absolute(map_mean)
    # Brainmask !!! without CSF and with erosion !!!
    map_mean[brainmask == 0] = 0

    # Set to 0 for better visualization
    #map_mean[map_mean < 0.2] = 0

    # !!! NO CSF !!!
    result_images(map_mean, name, mask, cmap="hot", note="hot")
    result_images(map_mean, name, mask, cmap="gray", note="gray")
    result_images(binary_opening(np.where(map_mean >= 0.5, 1, 0), structure=np.ones((3,3,3))).astype(int), name, mask, cmap="gray", note="opening")

    print("Save map: ", name)
    np.save(opj(config.results_path, 'map_') + name, map_mean)

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

        # print image
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
        map = np.where(map >= 0.5, 1, 0)
        fig, ax1 = plt.subplots(1, 1, figsize=(20, 20))
        ax1.imshow(montage(map), cmap=cmap)
        path = opj(config.results_path, 'image_map_' + name + note + '.png')
        fig.savefig(path)
        plt.close(fig)

    return


"""

CORRECT FUNKTION
TODO: !!

"""
def latentspace_analysis(space, brainmask, mask):
    mu_brain, mu_brain2, log_var_brain, mu_tumor, mu_tumor2, log_var_tumor = [], [], [], [], [], []

    for index, value in np.ndenumerate(brainmask):
        if value == 0:
            continue
        else:
            # Classify values: tumor or not
            if mask[index[0], index[1], index[2]] == 0:
                mu_brain.append(space[0, index[0], index[1], index[2]])
                mu_brain2.append(space[1, index[0], index[1], index[2]])
                log_var_brain.append(space[2, index[0], index[1], index[2]])
            else:
                mu_tumor.append(space[0, index[0], index[1], index[2]])
                mu_tumor2.append(space[1, index[0], index[1], index[2]])
                log_var_tumor.append(space[2, index[0], index[1], index[2]])

    mu_brain_arr = np.asarray(mu_brain)
    mu_brain2_arr = np.asarray(mu_brain2)
    log_var_brain_arr = np.asarray(log_var_brain)
    mu_tumor_arr = np.asarray(mu_tumor)
    mu_tumor2_arr = np.asarray(mu_tumor2)
    log_var_tumor_arr = np.asarray(log_var_tumor)

    from scipy.stats.kde import gaussian_kde
    from numpy import linspace

    # this create the kernel, given an array it will estimate the probability over that values
    kde_brain = gaussian_kde([x / len(mu_brain2) for x in mu_brain2])
    kde_tumor = gaussian_kde([x / len(mu_tumor2) for x in mu_tumor2])
    # these are the values over wich your kernel will be evaluated
    dist_space = linspace(min(mu_brain2), max(mu_brain2), 100)
    # plot the results
    plt.plot(dist_space, kde_brain(dist_space), color='blue', label='Healthy')
    plt.plot(dist_space, kde_tumor(dist_space), color='red', label='Tumor')

    plt.legend()
    plt.xlabel("Mean D2")
    plt.ylabel("Propability")
    plt.show()
    plt.close()

    plt.scatter(mu_brain_arr, mu_brain2_arr, cmap='blue', label='Healthy')
    plt.scatter(mu_tumor_arr, mu_tumor2_arr, cmap='red', label='Tumor')

    # plt.colorbar()
    plt.legend()
    plt.xlabel("Mean D1")
    plt.ylabel("Mean D2")
    plt.show()
    plt.close()

    return


# Hier sollen die gespeicherten Bilder aus dem Netzwerk aufgerufen werden und anschließend bis zur map verarbeitet + gespeichert werden
def processing():
    input_path = config.test
    mask_path = config.test_mask
    # !!! NO CSF !!!
    brainmask_path = glob(opj('/work/scratch/ecke/Masterarbeit/Data/Test', 'brainmask_withoutCSF*'))
    brainmask_path.sort()
    latentspace_size = 2

    if (config.network == "VoxelVAE") or (config.network == "CNNVoxelVAE"):
        # Load all the data
        output_path = glob(opj(config.data_drop_off, "batch*"))
        output_path.sort()
        id_list = []
        latentspace_list = []

        # Create a numpy array with shape (64,64,80,64) for every available test image
        for path in input_path:
            path_id = path.split("/")[-1].replace(".npy", "")
            id_list.append(path_id)
            locals()[path_id] = np.zeros([64, 64, 80, 64])

            path_id_latent = path.split("/")[-1].replace(".npy", "_latentspace")
            latentspace_list.append(path_id_latent)
            locals()[path_id_latent] = np.zeros([2 * latentspace_size, 64, 80, 64])

        # Load test data from server and reconstruct all test images (results from network to compare with input)
        for batch in output_path[0:10]:

            with open(batch, "rb") as f:
                loaded_batch = pickle.load(f)
            print(batch)

            for index in range(len(loaded_batch[1])):
                # Define image of voxel
                voxel_id = loaded_batch[1][index].split("/")[-1].replace(".npy", "")
                voxel_id_latentspace = loaded_batch[1][index].split("/")[-1].replace(".npy", "_latentspace")

                # Define coordinates of voxel in image
                voxel_x = loaded_batch[2][0][index] - 1
                voxel_y = loaded_batch[2][1][index] - 1
                voxel_z = loaded_batch[2][2][index] - 1

                # Save voxel in image
                voxel = loaded_batch[0][index]
                mu = loaded_batch[3][index]
                log_var = loaded_batch[4][index]
                locals()[voxel_id][:, voxel_x, voxel_y, voxel_z] = voxel.cpu()
                locals()[voxel_id_latentspace][0:latentspace_size, voxel_x, voxel_y, voxel_z] = mu.cpu()
                locals()[voxel_id_latentspace][latentspace_size:2 * latentspace_size, voxel_x, voxel_y,
                voxel_z] = log_var.cpu()

        for i in range(len(id_list)):
            input = np.load(input_path[i])
            mask = np.load(mask_path[i])
            brainmask = np.load(brainmask_path[i])
            result = locals()[id_list[i]]
            latent_space = locals()[latentspace_list[i]]
            name = id_list[i]
            np.save(opj(config.results_path, 'raw_output_') + name, np.float32(result))

            # Postprocessing of Approach 3
            latentspace_analysis(latent_space, brainmask, mask)
            postprocessing_voxel(result, input, brainmask, name, mask)

        return

    output_path = glob(opj(config.results_path, "output_*"))
    output_path.sort()

    for i in range(len(input_path)):
        input = np.load(input_path[i])
        mask = np.load(mask_path[i])
        # !!! NO CSF and with erosion!!!
        brainmask = np.load(brainmask_path[i])
        output = np.load(output_path[i])
        name = input_path[i].split("/")[-1]
        print(name)

        if config.network == "VanillaVAE":
            postprocessing_baseline(output, input, brainmask, name, mask)
        if config.network == "RecDisc":
            postprocessing_reddisc(output, input, brainmask, name, mask)


    return
