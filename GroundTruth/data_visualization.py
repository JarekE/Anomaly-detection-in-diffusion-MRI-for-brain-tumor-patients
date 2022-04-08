# Idea: Load images, show them, save data-information for BraTS Pipeline

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob

"""

vpX
(1-36, without 15 = 35 subjects)

Data:
T1: (232, 256, 176)
Flair: (232, 256, 176)

x: Set-value: Coronal plane
y: Set-Value: Transverse plane
z: Set-value: Sagittal plane 

Direction in space is not "normal"


"""

imgs = [nib.load(f'/images/Diffusion_Imaging/uka_gliom/ma_ecke/vp{t}/{m}.nii.gz').get_fdata().astype(np.float32) for t in [x for x in range(1,37) if x != 15] for m in ["t1_skullstripped", "t2_flair_masked"]]
#lbl = nib.load("/data/BraTS2021_train/BraTS2021_00000/BraTS2021_00000_seg.nii.gz").get_fdata().astype(np.uint8)[:, :, 75]
fig, ax = plt.subplots(nrows=36*2, ncols=1, figsize=(10, 360))
for i, img in enumerate(imgs):
    ax[i].imshow(img[:, 170, :], cmap='gray')
    ax[i].axis('off')
#ax[-1].imshow(lbl, vmin=0, vmax=4)
#ax[-1].axis('off')
plt.tight_layout()
plt.show()
