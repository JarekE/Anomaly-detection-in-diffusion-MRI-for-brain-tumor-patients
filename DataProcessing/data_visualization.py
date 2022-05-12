# Idea: Load images, show them, save data-information for BraTS Pipeline

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
from glob import glob
from skimage.util import montage
import shutil


def preview_images():

    # Data
    imgs = [nib.load(f"/images/PublicDataset/brats/RSNA_ASNR_MICCAI_BraTS2021_TrainingData/BraTS2021_00000/BraTS2021_00000_{m}.nii.gz").get_fdata().astype(np.float32) for m in ["flair", "t1"]]
    imgs_uka = [nib.load(f'/images/Diffusion_Imaging/uka_gliom/ma_ecke/vp2/{m}.nii.gz').get_fdata().astype(np.float32)for m in ["t2_flair_masked", "t1_skullstripped"]]
    lbl = nib.load("/images/PublicDataset/brats/RSNA_ASNR_MICCAI_BraTS2021_TrainingData/BraTS2021_00000/BraTS2021_00000_seg.nii.gz").get_fdata().astype(np.uint8)

    # UKA Datapipeline Beta-version
    for i, img in enumerate(imgs_uka):
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 0, 1)
        imgs_uka[i] = img

    # Example: Show BraTS data
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 15))
    for i, img in enumerate(imgs):
        ax[0, i].imshow(img[:, :, 100], cmap='gray')
        ax[0, i].axis('off')
    ax[0, -1].imshow(lbl[:, :, 100], vmin=0, vmax=4)
    ax[0, -1].axis('off')

    for i, img in enumerate(imgs_uka):
        ax[1, i].imshow(img[:, :, 150], cmap='gray')
        ax[1, i].axis('off')
    ax[1, -1].axis('off')
    plt.tight_layout()
    plt.show()

    print("BraTS Shape: ", imgs[0].shape)
    print("UKA Shape: ", imgs_uka[0].shape)


def pipeline_data():

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    train = np.load("/work/scratch/ecke/Groundtruth_Data/train/BraTS2021_00000_x.npy")
    train_mask = np.load("/work/scratch/ecke/Groundtruth_Data/train/BraTS2021_00000_y.npy")

    b = 75
    ax[0].imshow(train[0, b, :, :], cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(train[1, b, :, :], cmap='gray')
    ax[1].axis('off')
    #ax[1].set_title("Training data (BraTS, pre-processed)", fontsize=40)
    ax[2].imshow(train_mask[0, b, :, :], vmin=0, vmax=4)
    ax[2].axis('off')
    plt.tight_layout()
    #plt.show()
    fig.savefig('train.png', dpi=fig.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    test = np.load("/work/scratch/ecke/Groundtruth_Data/test/vp3_x.npy")
    test_meta = np.load("/work/scratch/ecke/Groundtruth_Data/test/vp3_meta.npy")

    b = 100
    ax[0].imshow(test[0, b, :, :], cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(test[1, b, :, :], cmap='gray')
    ax[1].axis('off')
    #ax[1].set_title("Testing data (UKA, pre-processed)", fontsize=40)
    ax[2].axis('off')
    plt.tight_layout()
    #plt.show()
    fig.savefig('test.png', dpi=fig.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    test_ori = nib.load("/work/scratch/ecke/ma_ecke/images/vp3.nii.gz").get_fdata()
    pred = nib.load("/work/scratch/ecke/Groundtruth_Data/results/final_preds/vp3.nii.gz").get_fdata().astype(np.uint8)

    n = 200
    ax[0].imshow(test_ori[:, n, :, 0], cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(test_ori[:, n, :, 1], cmap='gray')
    ax[1].axis('off')
    #ax[1].set_title("Results (UKA)", fontsize=40)
    ax[2].imshow(pred[:, n, :], vmin=0, vmax=4)
    ax[2].axis('off')
    plt.tight_layout()
    #plt.show()
    fig.savefig('results.png', dpi=fig.dpi)
    plt.close(fig)

    print(test_meta)
    print("Training Data Shape: ", train.shape)
    print("Testing Data Shape: ", test.shape)
    print("Original Data Shape: ", test_ori.shape)
    print("Result Mask Shape: ", pred.shape)


def mask_control():

    all_images = glob(os.path.join('/work/scratch/ecke/ma_ecke_processed/images', 'vp*'))
    all_masks = glob(os.path.join('/work/scratch/ecke/Groundtruth_Data/results/final_preds', 'vp*'))
    if os.path.exists("final_preds"):
        shutil.rmtree("final_preds")
    os.mkdir("final_preds")

    for i in range(len(all_images)):
        test_image = nib.load(all_images[i]).get_fdata()
        test_mask = nib.load(all_masks[i]).get_fdata()
        example_id = all_images[i].split("/")[-1]

        test_image = np.swapaxes(test_image, 0, 1)
        test_mask = np.swapaxes(test_mask, 0, 1)

        # print image
        fig, ax1 = plt.subplots(1, 1, figsize=(20, 20))
        ax1.imshow(montage(test_image[... , 0]), cmap='gray')
        print("Information about: ", example_id)
        print("Maximal value in mask: ", np.amax(test_mask))
        print("Is 3 part of mask: ", 3 in test_mask)
        print("Affine matrix of image and mask are equal: ", np.array_equal(nib.load(all_masks[i]).affine, nib.load(all_images[i]).affine))
        test_mask[test_mask == 0] = np.nan
        ax1.imshow(montage(test_mask), vmin=0, vmax=4, alpha=1)
        path = os.path.join('final_preds', example_id + '_overlay.png')
        fig.savefig(path)
        plt.close(fig)


def tumor_diffspace_control():

    all_folder = glob('/images/Diffusion_Imaging/uka_gliom/ma_ecke/vp*')
    tumor_folder = [folder for folder in all_folder if folder[-1] != "l"]
    dir = "tumor_diffspace_control"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

    for i in range(len(tumor_folder)):
        id_image = tumor_folder[i] + "/dwi.nii.gz"
        id_mask = tumor_folder[i] + "/tumor_diffspace.nii.gz"
        test_image = nib.load(id_image).get_fdata()
        test_mask = nib.load(id_mask).get_fdata()
        example_id = tumor_folder[i].split("/")[-1]

        test_image = np.swapaxes(test_image, 0, 2)
        test_mask = np.swapaxes(test_mask, 0, 2)

        # print image
        fig, ax1 = plt.subplots(1, 1, figsize=(20, 20))
        ax1.imshow(montage(test_image[... , 0]), cmap='gray')
        # do not pint background values == 0
        test_mask[test_mask == 0] = np.nan
        ax1.imshow(montage(test_mask), vmin=0, vmax=4, alpha=1)
        path = os.path.join('tumor_diffspace_control', example_id + '_overlay.png')
        fig.savefig(path)
        plt.close(fig)


#mask_control()

def testfunc():
    print("yes!")

#tumor_diffspace_control()

"""
test = np.load("/work/scratch/ecke/Groundtruth_Data/test/vp3_x.npy")
test_meta = np.load("/work/scratch/ecke/Groundtruth_Data/test/vp3_meta.npy")

test2 = np.load("/work/scratch/ecke/Groundtruth_Data/test/vp1_x.npy")
test_meta2 = np.load("/work/scratch/ecke/Groundtruth_Data/test/vp1_meta.npy")

print("x")

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

test = np.load("/work/scratch/ecke/Groundtruth_Data/test/vp3_x.npy")
test_meta = np.load("/work/scratch/ecke/Groundtruth_Data/test/vp3_meta.npy")

b = 90
ax[0].imshow(test[0, b, :, :], cmap='gray')
ax[0].axis('off')
ax[1].imshow(test[1, b, :, :], cmap='gray')
ax[1].axis('off')
#ax[1].set_title("Testing data (UKA, pre-processed)", fontsize=40)
ax[2].axis('off')
plt.tight_layout()
#plt.show()
fig.savefig('test.png', dpi=fig.dpi)
plt.close(fig)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

test2 = np.load("/work/scratch/ecke/Groundtruth_Data/test/vp5_x.npy")
test_meta2 = np.load("/work/scratch/ecke/Groundtruth_Data/test/vp5_meta.npy")

b = 100
ax[0].imshow(test2[0, b, :, :], cmap='gray')
ax[0].axis('off')
ax[1].imshow(test2[1, b, :, :], cmap='gray')
ax[1].axis('off')
#ax[1].set_title("Testing data (UKA, pre-processed)", fontsize=40)
ax[2].axis('off')
plt.tight_layout()
#plt.show()
fig.savefig('test2.png', dpi=fig.dpi)
plt.close(fig)
"""