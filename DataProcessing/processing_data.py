# This file should process my data+masks.
# Goal: numpy files in Data/Train or /Test with cropped and "diffusion"-processed data ready for my networks!

import numpy as np
from glob import glob
from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from os.path import join as opj
from scipy import ndimage

# Get data
PATH = "/work/scratch/ecke/Unprocessed_Data"
train_list = glob(opj(PATH, 'Train', 'vp*'))
test_list = glob(opj(PATH, 'Test', 'vp*'))

# Save data
PATH_SAVE = "/work/scratch/ecke/Masterarbeit/Data"
train_path = opj(PATH_SAVE, 'Train')
test_path = opj(PATH_SAVE, 'Test')

# Healthy subjects from UKA Dataset. Used to train the anomaly models.
def process_train():

    for dataset in train_list:

        example_id = dataset.split("/")[-1]
        b_name = example_id.replace("ctrl","_ctrl_")
        print(example_id)

        brainmask, _ = load_nifti(opj(dataset, "b0_brainmask.nii.gz"))
        dwi, aff = load_nifti(opj(dataset, "dwi.nii.gz"))

        dwi = dwi * np.expand_dims(brainmask, axis=-1)

        bvals, bvecs = read_bvals_bvecs(opj(dataset, b_name+"bvals"), opj(dataset, b_name+"bvecs"))
        bvals = np.around(bvals / 1000).astype(np.int) * 1000
        print(bvals)

        # scale the b-values between 1 and 0 (Diffusionsabschwächung)
        meanb0 = np.expand_dims(np.mean(dwi[..., bvals < 150], axis=-1), axis=-1)
        edw = np.divide(dwi, meanb0)
        edw[edw > 1] = 1
        edw[edw < 0] = 0
        edw[brainmask == 0] = 0
        edw[np.isnan(edw)] = 0

        # delete b0 values
        dwi = np.delete(edw, np.where(bvals == 0), axis=3)

        # move axis
        dwi = np.moveaxis(dwi, 3, 0)
        print(dwi.shape)

        # crop to 64, 80, 56
        ch, x, y, z = dwi.shape
        x1 = x // 2 - 32
        x2 = x // 2 + 32
        y1 = y // 2 - 40
        y2 = y // 2 + 40

        patch_dwi = dwi[:, x1:x2, y1:y2, :]
        patch_dwi = np.float32(np.concatenate((np.zeros((*patch_dwi.shape[:3], 5)), patch_dwi,
                                               np.zeros((*patch_dwi.shape[:3], 5))), axis=-1))
        patch_brainmask = brainmask[x1:x2, y1:y2, :]
        patch_brainmask = np.float32(np.concatenate((np.zeros((*patch_brainmask.shape[:2], 5)), patch_brainmask,
                                               np.zeros((*patch_brainmask.shape[:2], 5))), axis=-1))

        np.save(opj(train_path, example_id), patch_dwi)
        np.save(opj(train_path, 'b0_brainmask_'+example_id), patch_brainmask)


# Tumor data + masks.
def process_test():

    for dataset in test_list:

        example_id = dataset.split("/")[-1]
        print(example_id)

        brainmask, _ = load_nifti(opj(dataset, "b0_brainmask.nii.gz"))
        dwi, aff = load_nifti(opj(dataset, "dwi.nii.gz"))
        mask, _ = load_nifti(opj(dataset, "tumor_diffspace.nii.gz"))
        segmentation, _ = load_nifti(opj(dataset, "seg_diffspace.nii.gz"))

        dwi = dwi * np.expand_dims(brainmask, axis=-1)

        bvals, bvecs = read_bvals_bvecs(opj(dataset, "bvals"), opj(dataset, "bvecs"))
        bvals = np.around(bvals / 1000).astype(np.int) * 1000
        #print(bvals)

        # scale the b-values between 1 and 0 (Diffusionsabschwächung)
        meanb0 = np.expand_dims(np.mean(dwi[..., bvals < 150], axis=-1), axis=-1)
        edw = np.divide(dwi, meanb0)
        edw[edw > 1] = 1
        edw[edw < 0] = 0
        edw[brainmask == 0] = 0
        edw[np.isnan(edw)] = 0

        # delete b0 values
        dwi = np.delete(edw, np.where(bvals == 0), axis=3)

        # move axis
        dwi = np.moveaxis(dwi, 3, 0)

        # crop to 64, 80, 64
        ch, x, y, z = dwi.shape
        x1 = x // 2 - 32
        x2 = x // 2 + 32
        y1 = y // 2 - 40
        y2 = y // 2 + 40

        patch_dwi = dwi[:, x1:x2, y1:y2, :]
        patch_mask = mask[x1:x2, y1:y2, :]
        patch_brainmask = brainmask[x1:x2, y1:y2, :]
        patch_segmentation = segmentation[x1:x2, y1:y2, :]

        if brainmask.shape[2] == 62:
            sub = 4
        else:
            sub = 0
        patch_dwi = np.float32(np.concatenate((np.zeros((*patch_dwi.shape[:3], (5-sub))), patch_dwi,
                                               np.zeros((*patch_dwi.shape[:3], (5-sub)))), axis=-1))
        patch_mask = np.float32(np.concatenate((np.zeros((*patch_mask.shape[:2], (5-sub))), patch_mask,
                                                np.zeros((*patch_mask.shape[:2], (5-sub)))), axis=-1))
        patch_brainmask = np.float32(np.concatenate((np.zeros((*patch_brainmask.shape[:2],(5-sub))), patch_brainmask,
                                                np.zeros((*patch_brainmask.shape[:2], (5-sub)))), axis=-1))
        patch_segmentation = np.float32(np.concatenate((np.zeros((*patch_segmentation.shape[:2], (5 - sub))), patch_segmentation,
                                                     np.zeros((*patch_segmentation.shape[:2], (5 - sub)))), axis=-1))

        # Create a white matter only mask from the segmentation mask
        new_whitemask = np.float32(np.where(patch_segmentation == 3, 1, 0))

        # Erosion of the edited brainmask
        brainmask_edited = ndimage.binary_erosion(patch_brainmask, structure=np.ones((2,2,2))).astype(patch_brainmask.dtype)
        # Create a brainmask without csf
        brainmask_without_csf = np.float32(np.where(patch_segmentation == 1, 0, brainmask_edited))

        np.save(opj(test_path, example_id), patch_dwi)
        np.save(opj(test_path, "mask"+example_id), patch_mask)
        np.save(opj(test_path, "whitemask" + example_id), new_whitemask)
        np.save(opj(test_path, "b0_brainmask" + example_id), patch_brainmask)
        np.save(opj(test_path, "brainmask_withoutCSF" + example_id), brainmask_without_csf)


def correction_uka_mask():
    PATH_MASK = "/work/scratch/ecke/Masterarbeit/Data/Test"
    mask_list = [opj(PATH_MASK, "maskvp3.npy"), opj(PATH_MASK, "maskvp17.npy"), opj(PATH_MASK, "maskvp18.npy"), opj(PATH_MASK, "maskvp27.npy")]

    for m in mask_list:
        mask = np.load(opj(PATH_MASK, m))
        mask = np.where(mask != 0, 4, 0)
        np.save(opj(PATH_MASK, m), mask.astype(int))


# Need new data? Lets go!
"""
process_train()
print("Process: Test")
process_test()
correction_uka_mask()
"""
process_test()