# Everything about post-processing
import os
import shutil
from glob import glob
from os.path import join as opj
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import montage
from scipy import ndimage
from skimage import filters

import config

def prepare_results():
  path = config.save_path
  if os.path.exists(config.results_path):
    shutil.rmtree(config.results_path)
  os.mkdir(config.results_path)

  return path


def result_images(map, name, mask):
  mask_dil = ndimage.binary_dilation(mask, iterations=2)
  mask_contour = mask_dil - mask

  # Rotate image (not data!) for optimal visual output
  mask_contour = np.flip(np.flip(np.swapaxes(mask_contour, 0, 2), axis=0), axis=1)
  map = np.flip(np.flip(np.swapaxes(map, 0, 2), axis=0), axis=1)

  # print image
  fig, ax1 = plt.subplots(1, 1, figsize=(20, 20))
  ax1.imshow(montage(map), cmap='inferno')
  mask_contour = mask_contour.astype(float)
  mask_contour[mask_contour > 0.5] = 1
  mask_contour[mask_contour <= 0.5] = np.nan
  ax1.imshow(montage(mask_contour), 'binary', alpha=1)
  path = opj(config.results_path, 'image_map_'+name+'.png')
  fig.savefig(path)
  plt.close(fig)


# Hier sollen die gespeicherten Bilder aus dem Netzwerk aufgerufen werden und anschließend bis zur map verarbeitet + gespeicehrt werden
def processing():
  input_path = config.test
  mask_path = config.test_mask
  brainmask_path = config.test_b0_brainmask
  output_path = glob(opj(config.results_path, "output_*"))
  output_path.sort()

  for i in range(len(input_path)):
    input = np.load(input_path[i])
    mask = np.load(mask_path[i])
    brainmask = np.load(brainmask_path[i])
    output = np.load(output_path[i])
    name = input_path[i].split("/")[-1]

    # Postprocessing v1.0
    output /= output.max()
    map = np.subtract(input, output)
    map_mean = np.mean(map, axis=0)
    map_abs = np.absolute(map_mean)
    map_abs[brainmask == 0] = 0

    if input_path[i] == '/work/scratch/ecke/Masterarbeit/Data/Test/vp1.npy':
      plt.figure()
      plt.imshow(input[0, :, :, 20], cmap='gray')
      plt.show()
      plt.close()

      plt.figure()
      plt.imshow(map_abs[:, :, 20], cmap='inferno')
      plt.show()
      plt.close()

    result_images(map_abs, name, mask)
    print("Save map: ", name)
    np.save(opj(config.results_path, 'map_')+name, map_abs)

#processing()