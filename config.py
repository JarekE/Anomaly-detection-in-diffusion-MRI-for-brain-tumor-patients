# Define my configs.
from glob import glob
from os.path import join as opj
import random
from typing import TypeVar
from argparse import ArgumentParser


# For full images, we have only 28. 4 is a power of 2 and a divider of 28.
batch_size = 4
latent_dim = 256

# VanillaVAE params
vanilla_params = {"LR": 0.005,
  "weight_decay": 0.0,
  "scheduler_gamma": 0.95,
  "kld_weight": 0.0000122,      # input.shape.flatten / latent space dimensions
  "manual_seed": 1265}

unet_params = {"LR": 0.001,
  "weight_decay": 0.0,
  "scheduler_gamma": None,
  "kld_weight": False,      # input.shape.flatten / latent space dimensions
  "manual_seed": 1265}

# Arguments
parser = ArgumentParser()
parser.add_argument("-m", "--mode", default="train", help="Training mode of network (train or test)")
parser.add_argument("-t", "--test_this_model", default="VanillaVAE-epoch=87-val_loss=5.81-max_epochs=3000.ckpt", help="Name of model to be tested. Only usefull if mode is -test-.")
parser.add_argument("-e", "--epochs", default=200, type=int, help="Number of epochs to train the model.")
parser.add_argument("-n", "--network", default="VanillaVAE", help="Name of Network: VanillaVAE or UNet.")
parser.add_argument("-a", "--augmentation", default=False, type=bool, help="Should the training data be augmented by applying transformations to it?")
args = vars(parser.parse_args())
mode = args["mode"]
test_this_model = args["test_this_model"]
epochs = args["epochs"]
network = args["network"]
augmentation = args["augmentation"]

# Load data. Training data is randomly shuffled.
img_path_uka = '/work/scratch/ecke/Masterarbeit/Data'
train = glob(opj(img_path_uka, "Train", "vp*"))
random.shuffle(train)
test = glob(opj(img_path_uka, "Test", "vp*"))
test.sort()
test_mask = glob(opj(img_path_uka, "Test", "mask*"))
test_mask.sort()
test_b0_brainmask = glob(opj(img_path_uka, "Test", "b0_brainmask*"))
test_b0_brainmask.sort()
uka_subjects = {"training": train[0:24], "validation": train[24:28], "test": test[0:32], "test_mask": test[0:32]}

# Names and paths
Tensor = TypeVar('torch.tensor')
log_dir = "/work/scratch/ecke/Masterarbeit/logs/Callback"
log_dir_logger = "/work/scratch/ecke/Masterarbeit/logs/Logger"
save_path = opj("/work/scratch/ecke/Masterarbeit/logs/Callback", test_this_model)
results_path = opj("/work/scratch/ecke/Masterarbeit/Results", test_this_model)
file_name = network+'-{epoch:02d}-{val_loss:.2f}-max_epochs='+str(epochs)
