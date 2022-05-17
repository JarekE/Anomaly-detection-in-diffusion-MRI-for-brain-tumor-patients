# Everything about post-processing
import os
import shutil
from glob import glob
from os.path import join as opj
import matplotlib.pyplot as plt
import numpy as np

import config

def prepare_results():
  path = config.save_path
  if os.path.exists(config.results_path):
    shutil.rmtree(config.results_path)
  os.mkdir(config.results_path)

  return path

# Hier sollen die gespeicherten Bilder aus dem Netzwerk aufgerufen werden und anschließend bis zur map verarbeitet + gespeicehrt werden
def processing():
  input_path = config.test
  output_path = glob(opj(config.results_path, "output_*"))
  output_path.sort()

  for i in range(len(input_path)):
    input = np.load(input_path[i])
    output = np.load(output_path[i])
    name = input_path[i].split("/")[-1]

    # Postprocessing v1.0
    output /= output.max()
    map = np.subtract(input, output)
    map_mean = np.mean(map, axis=0)
    map_abs = np.absolute(map_mean)

    if input_path[i] == '/work/scratch/ecke/Masterarbeit/Data/Test/vp1.npy':
      plt.figure()
      plt.imshow(input[0, :, :, 20], cmap='gray')
      plt.show()
      plt.close()

      plt.figure()
      plt.imshow(map_abs[:, :, 20], cmap='inferno')
      plt.show()
      plt.close()

    np.save(opj(config.results_path, 'map_')+name, map_abs)

processing()