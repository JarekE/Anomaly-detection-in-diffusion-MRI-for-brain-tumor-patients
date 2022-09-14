# Idea: Load images, show them, save data-information for BraTS Pipeline

import numpy as np
import itertools
import matplotlib.pyplot as plt
import nibabel as nib
import os
from glob import glob
from skimage.util import montage
import shutil
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import config

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


def final_data_test():

    data_folder = sorted(glob('/work/scratch/ecke/Masterarbeit/Data/Test/vp*'))
    mask_folder = sorted(glob('/work/scratch/ecke/Masterarbeit/Data/Test/mask*'))
    dir = "final_data_test"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

    for i in range(len(data_folder)):

        test_image = np.load(data_folder[i])
        test_mask = np.load(mask_folder[i])
        example_id = data_folder[i].split("/")[-1]

        #test_image = np.swapaxes(test_image, 1, 3)
        #test_mask = np.swapaxes(test_mask, 0, 2)
        test_image = np.flip(np.flip(np.swapaxes(test_image, 1, 3), axis=1), axis=2)
        test_mask = np.flip(np.flip(np.swapaxes(test_mask, 0, 2), axis=0), axis=1)

        """
        if example_id == "vp8.npy":
            test_image = np.swapaxes(test_image, 1, 3)
            test_mask = np.swapaxes(test_mask, 0, 2)
            np.save(opj("/work/scratch/ecke/Masterarbeit/Data/Test", example_id), test_image)
            np.save(opj("/work/scratch/ecke/Masterarbeit/Data/Test", "mask" + example_id), test_mask)
            
        print(example_id)
        print(test_image.shape)
        print(test_mask.shape)    
        """

        # print image
        fig, ax1 = plt.subplots(1, 1, figsize=(20, 20))
        # mean
        ax1.imshow(montage(np.mean(test_image, axis=0)[:, :, :]), cmap='gray')
        # do not pint background values == 0
        test_mask = test_mask.astype(float)
        test_mask[test_mask == 0] = np.nan
        ax1.imshow(montage(test_mask), vmin=0, vmax=4, alpha=1)
        path = os.path.join(dir, example_id.replace(".npy","") + '_overlay.png')
        fig.savefig(path)
        plt.close(fig)


def final_data_train():

    data_folder = sorted(glob('/work/scratch/ecke/Masterarbeit/Data/Train/vp*'))
    dir = "final_data_train"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

    for i in range(len(data_folder)):

        test_image = np.load(data_folder[i])
        example_id = data_folder[i].split("/")[-1]

        #test_image = np.swapaxes(test_image, 1, 3)
        test_image = np.flip(np.flip(np.swapaxes(test_image, 1, 3), axis=1), axis=2)

        # print image
        fig, ax1 = plt.subplots(1, 1, figsize=(20, 20))
        # Use first bvec-image (random, can be other)
        ax1.imshow(montage(test_image[0, ...]), cmap='gray')
        path = os.path.join(dir, example_id.replace(".npy","") + '_overlay.png')
        fig.savefig(path)
        plt.close(fig)

def show_RecDisc(input, input_anomaly, reconstructive_map, results, print_value, reconstruction):

    fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(10, 5))
    ax[0,0].imshow(input.detach().cpu().numpy()[2, 30, :, :, print_value.detach().cpu().numpy()[2]], cmap='gray')
    ax[0,0].axis('off')
    ax[0,0].title.set_text("Data 30/64")
    ax[0,1].imshow(input_anomaly.detach().cpu().numpy()[2, 30, :, :, print_value.detach().cpu().numpy()[2]], cmap='gray')
    ax[0,1].axis('off')
    ax[0,1].title.set_text("Input 30/64")
    ax[0,2].imshow(reconstruction.detach().cpu().numpy()[2, 30, :, :, print_value.detach().cpu().numpy()[2]], cmap='gray')
    ax[0,2].axis('off')
    ax[0,2].title.set_text("Reconstruction 30/64")
    ax[0,3].imshow(reconstructive_map.detach().cpu().numpy()[2, 0, :, :, print_value.detach().cpu().numpy()[2]], cmap='gray')
    ax[0,3].axis('off')
    ax[0,3].title.set_text("GroundTruth")
    ax[0,4].imshow(results.detach().cpu().numpy()[2, 0, :, :, print_value.detach().cpu().numpy()[2]], cmap='gray')
    ax[0,4].axis('off')
    ax[0,4].title.set_text("Result")
    ax[0,5].imshow(np.where(results.detach().cpu().numpy() > 0.5, 1, 0)[2, 0, :, :, print_value.detach().cpu().numpy()[2]], cmap='gray')
    ax[0,5].axis('off')
    ax[0,5].title.set_text("0.5")
    ax[1,0].imshow(np.mean(input.detach().cpu().numpy(), axis=1)[2, :, :, print_value.detach().cpu().numpy()[2]], cmap='gray')
    ax[1,0].axis('off')
    ax[1,0].title.set_text("Data Mean")
    ax[1,1].imshow(np.mean(input_anomaly.detach().cpu().numpy(), axis=1)[2, :, :, print_value.detach().cpu().numpy()[2]], cmap='gray')
    ax[1,1].axis('off')
    ax[1,1].title.set_text("Input Mean")
    ax[1,2].imshow(np.mean(reconstruction.detach().cpu().numpy(), axis=1)[2, :, :, print_value.detach().cpu().numpy()[2]], cmap='gray')
    ax[1,2].axis('off')
    ax[1,2].title.set_text("Reconstruction Mean")
    ax[1,3].imshow(reconstructive_map.detach().cpu().numpy()[2, 0, :, :, print_value.detach().cpu().numpy()[2]], cmap='gray')
    ax[1,3].axis('off')
    ax[1,3].title.set_text("GroundTruth")
    ax[1,4].imshow(results.detach().cpu().numpy()[2, 0, :, :, print_value.detach().cpu().numpy()[2]], cmap='gray')
    ax[1,4].axis('off')
    ax[1,4].title.set_text("Result")
    ax[1,5].imshow(np.where(results.detach().cpu().numpy() > 0.5, 1, 0)[2, 0, :, :, print_value.detach().cpu().numpy()[2]], cmap='gray')
    ax[1,5].axis('off')
    ax[1,5].title.set_text("0.5")
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    return

def z_space_visualization(z_space, classes):
    if z_space.shape[0] > 1000:
        z_space = z_space[0:1000, :]
        classes = classes[0:1000]

    brain = z_space[classes == 0]
    anomaly = z_space[classes == 1]

    plt.scatter(brain[:, 0], brain[:, 1], c='b', label='Brain')
    plt.scatter(anomaly[:, 0], anomaly[:, 1], c='r', label='Anomaly')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend()
    plt.show()
    plt.close()
    return

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def z_space_analysis(z_space, classes):
    gmm = GaussianMixture(n_components=2).fit(z_space)
    gmm_labels = gmm.predict(z_space)

    # Confusion matrix
    cm = confusion_matrix(classes, gmm_labels)
    plot_confusion_matrix(cm, ['Brain matter', 'Anomaly'], title='z-Space Confusion Matrix', cmap='summer', normalize=True)

    # Plot in 2D (print only in 2D case)
    if config.latent_dim == 2:
        plt.scatter(z_space[:, 0], z_space[:, 1], c=gmm_labels, s=40, cmap='viridis')
        plt.show()
        plt.close()
    return

#final_data_test()
#final_data_train()
#tumor_diffspace_control()

"""
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

test = np.load("/work/scratch/ecke/Masterarbeit/Data/Test/maskvp1.npy")
test_brainmask = np.load("/work/scratch/ecke/Masterarbeit/Data/Test/brainmask_withoutCSFvp1.npy")

b = 35
ax[0].imshow(test[:, :, b], cmap='gray') 
ax[0].axis('off')
ax[1].imshow(test_brainmask[:, :, b], cmap='gray')
ax[1].axis('off')
plt.tight_layout()
plt.show()
#fig.savefig('test.png', dpi=fig.dpi)
plt.close(fig)
"""