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
import pickle

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
  return


# Hier sollen die gespeicherten Bilder aus dem Netzwerk aufgerufen werden und anschließend bis zur map verarbeitet + gespeicehrt werden
def processing():

  input_path = config.test
  mask_path = config.test_mask
  brainmask_path = config.test_b0_brainmask

  if (config.network == "VoxelVAE"):
    # Load all the data
    output_path = glob(opj('logs/DataDropOff', "batch*"))
    output_path.sort()
    id_list = []

    # Create a numpy array with shape (64,64,80,64) for every available test image
    for path in input_path:
      path_id = path.split("/")[-1].replace(".npy","")
      id_list.append(path_id)
      locals()[path_id] = np.empty([64,64,80,64])

    # Load test data from server and reconstruct all test images (results from network to compare with input)
    for batch in output_path:

      with open(batch, "rb") as f:
        loaded_batch = pickle.load(f)
      print(batch)

      for index in range(len(loaded_batch[2])):
        # Define image of voxel
        voxel_id = loaded_batch[2][index].split("/")[-1].replace(".npy","")

        # Define coordinates of voxel in image
        voxel_x = loaded_batch[3][0][index]
        voxel_y = loaded_batch[3][1][index]
        voxel_z = loaded_batch[3][2][index]

        # Save voxel in image
        voxel = loaded_batch[1][index]
        locals()[voxel_id][:, voxel_x, voxel_y, voxel_z] = voxel

        # mu_values = loaded_batch[4][index]
        # var_values = loaded_batch[5][index]

    for i in range(len(id_list)):
      input = np.load(input_path[i])
      mask = np.load(mask_path[i])
      brainmask = np.load(brainmask_path[i])
      result = locals()[id_list[i]]
      name = id_list[i]
      np.save(opj(config.results_path, 'raw_output_') + name, np.float32(result))

      plt.figure()
      plt.imshow(input[0, :, :, 20], cmap='gray')
      plt.show()
      plt.close()

      plt.figure()
      plt.imshow(result[0, :, :, 20], cmap='gray')
      plt.show()
      plt.close()

      # Postprocessing v1.0
      result /= result.max()
      map = np.subtract(input, result)
      map_mean = np.mean(map, axis=0)
      # !!! Information is lost !!! -> maybe change later
      map_abs = np.absolute(map_mean)
      # Clean the image
      map_abs[brainmask == 0] = 0
      # Dump the image
      map_abs[map_abs < 0.1] = 0

      result_images(map_abs, name, mask)
      print("Save map: ", name)
      np.save(opj(config.results_path, 'map_') + name, map_abs)

    return

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
    # !!! Information is lost !!! -> maybe change later
    map_abs = np.absolute(map_mean)
    # Clean the image
    map_abs[brainmask == 0] = 0
    # Dump the image
    map_abs[map_abs < 0.1] = 0

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
    return

processing()