# GroundTruth pipeline

"""
Header: Deep Learning segmentation of T1 / FLAIR images for dMRI ground truth maps of tumors

Data:
UKA (Testdata)
vpX, 35 subjects, T1/FLAIR, (232, 256, 176)
/images/Diffusion_Imaging/uka_gliom/ma_ecke/vp{t}/{m}.nii.gz

BraTS (Trainingsdata)
1251 subjects, not continous, T1/FLAIR (not the other!), (240, 240, 155)
/images/PublicDataset/brats/RSNA_ASNR_MICCAI_BraTS2021_TrainingData

Goal of this script:
Prepare both datasets, so that the nnUNet can process them as training and testing data
(calculate the ground thruth for the UKA data)

Important information:
the nnUNet project from NVIDIA is necessary for this script.
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/nnUNet/notebooks/BraTS21.ipynb
"""

import json
import os
from glob import glob
from subprocess import call
import nibabel
from joblib import Parallel, delayed
import shutil
import pickle
import os
from glob import glob
from subprocess import call
import nibabel as nib
import numpy as np
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt



# Location of the datasets. Important: The data is not there anymore. Please safe your own datasets.
# brats: .../BraTS*/... (all 1251 subjects)
# uka: .../vp*/... (only the tumor subjects)
brats_loc = "/images/PublicDataset/brats/RSNA_ASNR_MICCAI_BraTS2021_TrainingData"
uka_loc = "/work/scratch/ecke/ma_ecke"
uka_data = False


def load_nifty(directory, example_id, suffix):
    if uka_data:
        return nibabel.load(os.path.join(directory, suffix + ".nii.gz"))
    else:
        return nibabel.load(os.path.join(directory, example_id + "_" + suffix + ".nii.gz"))


def load_channels(d, example_id):
    if uka_data:
        return [load_nifty(d, example_id, data_type) for data_type in ["t2_flair_masked", "t1_skullstripped"]]
    else:
        return [load_nifty(d, example_id, suffix) for suffix in ["flair", "t1", "t1ce", "t2"]]


def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)


def prepare_nifty(d):
    example_id = d.split("/")[-1]
    print("Processing: ", example_id)
    if uka_data:
        flair, t1 = load_channels(d, example_id)
        affine, header = flair.affine, flair.header
        vol = np.stack([get_data(flair), get_data(t1)], axis=-1)
    else:
        flair, t1, t1ce, t2 = load_channels(d, example_id)
        affine, header = flair.affine, flair.header
        vol = np.stack([get_data(flair), get_data(t1), get_data(t1ce), get_data(t2)], axis=-1)

    vol = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
    nibabel.save(vol, os.path.join(d, example_id + ".nii.gz"))

    if os.path.exists(os.path.join(d, example_id + "_seg.nii.gz")):
        seg = load_nifty(d, example_id, "seg")
        affine, header = seg.affine, seg.header
        vol = get_data(seg, "unit8")
        vol[vol == 4] = 3
        seg = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
        nibabel.save(seg, os.path.join(d, example_id + "_seg.nii.gz"))


def prepare_dirs(data, train):
    img_path, lbl_path = os.path.join(data, "images"), os.path.join(data, "labels")
    call(f"mkdir {img_path}", shell=True)
    if train:
        call(f"mkdir {lbl_path}", shell=True)

    if uka_data:
        dirs = glob(os.path.join(data, "vp*"))
    else:
        dirs = glob(os.path.join(data, "BraTS*"))

    for d in dirs:
        if uka_data:
            files = glob(os.path.join(d, "*"))
            for f in files:
                if "t2_flair_masked" in f or "t1_skullstripped" in f:
                    continue
                else:
                    call(f"mv {f} {img_path}", shell=True)
            call(f"rm -rf {d}", shell=True)

        else:
            if "_" in d.split("/")[-1]:
                files = glob(os.path.join(d, "*.nii.gz"))
                for f in files:
                    if "flair" in f or "t1" in f or "t1ce" in f or "t2" in f:
                        continue
                    if "_seg" in f:
                        call(f"mv {f} {lbl_path}", shell=True)
                    else:
                        call(f"mv {f} {img_path}", shell=True)
            call(f"rm -rf {d}", shell=True)


def prepare_dataset_json(data, train):
    images, labels = glob(os.path.join(data, "images", "*")), glob(os.path.join(data, "labels", "*"))
    images = sorted([img.replace(data + "/", "") for img in images])
    labels = sorted([lbl.replace(data + "/", "") for lbl in labels])
    if uka_data:
        modality = {"0": "FLAIR", "1": "T1"}
    else:
        modality = {"0": "FLAIR", "1": "T1", "2": "T1CE", "3": "T2"}
    labels_dict = {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"}
    if train:
        key = "training"
        data_pairs = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
        key = "test"
        data_pairs = [{"image": img} for img in images]

    dataset = {
        "labels": labels_dict,
        "modality": modality,
        key: data_pairs,
    }

    with open(os.path.join(data, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)


def run_parallel(func, args):
    return Parallel(n_jobs=os.cpu_count())(delayed(func)(arg) for arg in args)


def prepare_dataset(data, train):
    print(f"Preparing Dataset from: {data}")
    if uka_data:
        run_parallel(prepare_nifty, sorted(glob(os.path.join(data, "vp*"))))
    else:
        run_parallel(prepare_nifty, sorted(glob(os.path.join(data, "BraTS*"))))
    prepare_dirs(data, train)
    prepare_dataset_json(data, train)


def prepare_data():
    if uka_data:
        # Testdata
        prepare_dataset(uka_loc, False)
    else:
        # Trainsdata
        prepare_dataset(brats_loc, True)
    print("Finished!")


def process_data(d):
    example_id = d.split("/")[-1]
    if example_id[0] == "v":
        image = np.load(d)
        if (example_id[5] == "x" or example_id[4] == "x"):
            image = np.swapaxes(image, 2, 3)
            image = np.swapaxes(image, 1, 3)
        else:
            for i in range(0, 4):
                image[i, [0, 1, 2]] = image[i, [1, 2, 0]]
        np.save(os.path.join("/work/scratch/ecke/Groundtruth_Data/test", example_id), image)
    else:
        image = np.load(d)
        print("Processing: ", example_id)
        if example_id[16] == "x":
            image = np.delete(image, 3, axis=0)
            image = np.delete(image, 2, axis=0)
        np.save(os.path.join("/work/scratch/ecke/Groundtruth_Data/train", example_id), image)


def save_data(data="uka"):
    if data == "uka":
        path = "/work/scratch/ecke/ma_ecke/12_3d/test"
        run_parallel(process_data, sorted(glob(os.path.join(path, "vp*"))))
    else:
        path = "/images/PublicDataset/brats/RSNA_ASNR_MICCAI_BraTS2021_TrainingData/11_3d"
        run_parallel(process_data, sorted(glob(os.path.join(path, "BraTS*"))))


def delete_images(path):
    list = sorted(glob(os.path.join(path, "BraTS*")))
    for image in list:
        example_id = image.split("/")[-1]
        if example_id[16] == "m":
            os.remove(image)
        else:
            continue

def change_config_file():
    data = pickle.load(open("/work/scratch/ecke/Groundtruth_Data/train/config.pkl", "rb"))
    data["in_channels"] = 3
    pickle.dump(data, open("/work/scratch/ecke/Groundtruth_Data/train/config.pkl", "wb"))


def to_lbl(pred):
    enh = pred[2]
    c1, c2, c3 = pred[0] > 0.5, pred[1] > 0.5, pred[2] > 0.5
    pred = (c1 > 0).astype(np.uint8)
    pred[(c2 == False) * (c1 == True)] = 2
    pred[(c3 == True) * (c1 == True)] = 4

    components, n = label(pred == 4)
    for et_idx in range(1, n + 1):
        _, counts = np.unique(pred[components == et_idx], return_counts=True)
        if 1 < counts[0] and counts[0] < 8 and np.mean(enh[components == et_idx]) < 0.9:
            pred[components == et_idx] = 1

    et = pred == 4
    if 0 < et.sum() and et.sum() < 73 and np.mean(enh[et]) < 0.9:
        pred[et] = 1

    pred = np.transpose(pred, (2, 1, 0)).astype(np.uint8)
    return pred


def prepare_preditions(e):
    fname = e[0].split("/")[-1].split(".")[0]
    preds = [np.load(f) for f in e]
    p = to_lbl(np.mean(preds, 0))
    p = np.swapaxes(p, 1, 2)
    p = np.swapaxes(p, 0, 2)

    img = nib.load(f"/work/scratch/ecke/ma_ecke/images/{fname}.nii.gz")
    nib.save(
        nib.Nifti1Image(p, img.affine, header=img.header),
        os.path.join("/work/scratch/ecke/Groundtruth_Data/results/final_preds", fname + ".nii.gz"),
    )


"""
------------------------------------------
# Call the pre-processing from NVIDIA (not necessary anymore in my pipeline)
------------------------------------------
# Prepare data for dataloader

# Set True, if uka data should be prepared
# uka_data = True

prepare_data()


# Call the prepocessing from NVIDIA

os.system("python /work/scratch/ecke/nnUNet/preprocess.py --task 11 --ohe --exec_mode training")
os.system("python /work/scratch/ecke/nnUNet/preprocess.py --task 12 --ohe --exec_mode test")
print("Finished!")
"""


"""
------------------------------------------
# Call the training and Inference from NVIDIA (start here for new training and/or new testing)
(Correct data is saved in specified folders)

Important:
- Use GRAM of >= 16GB (e.g. server or PC38)
------------------------------------------

# Training
ain.py --brats --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs XXX --fold 0 --amp --gpus 1 --task 11 --save_ckpt

# Testing (nnUnet.py)
main.py --gpus 1 --amp --save_preds --exec_mode predict --brats --data /work/scratch/ecke/Groundtruth_Data/test --ckpt_path /work/scratch/ecke/Groundtruth_Data/results/checkpoints/XXXXXXXXX.ckpt --tta

"""


"""
------------------------------------------
# Call the post-processing from NVIDIA
------------------------------------------
"""
os.makedirs("/work/scratch/ecke/Groundtruth_Data/results/final_preds")
preds = sorted(glob(f"/work/scratch/ecke/Groundtruth_Data/results/predictions_epoch=303*"))
examples = list(zip(*[sorted(glob(f"{p}/*.npy")) for p in preds]))
print("Preparing final predictions")
for e in examples:
    prepare_preditions(e)
print("Finished!")



"""
------------------------------------------
# Visualization
------------------------------------------
from GroundTruth import data_visualization as dv
dv.mask_control()
"""
