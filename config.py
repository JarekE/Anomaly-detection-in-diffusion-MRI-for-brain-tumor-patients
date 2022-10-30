from glob import glob
from os.path import join as opj
import random
from typing import TypeVar
from argparse import ArgumentParser 


# Arguments
parser = ArgumentParser()
parser.add_argument("-m", "--mode", default="train", help="Training mode of network (train or test)")
parser.add_argument("-t", "--test_this_model", default="RecDisc-epoch=107-val_loss=0.11-r=4-an=Mix-d=Half.ckpt", help="Name of model to be tested. Only usefull if mode is -test-.")
parser.add_argument("-e", "--epochs", default=250, type=int, help="Number of epochs to train the model.")
parser.add_argument("-n", "--network", default="RecDisc", help="Name of Network: VanillaVAE, UNet or RecDisc")
parser.add_argument("-l", "--latent_dim", default=2, type=int, help="Dimension of latent space, only relevant for VanillaVAE")
parser.add_argument("-r", "--run", default=8, type=int, help="Define the k-fold cross validation number (1-7). Use 8 for shuffle.")
parser.add_argument("-f", "--filter", default=128, type=int, help="Number of filters used in UNet or the RecDiscNet for reconstruction.")
parser.add_argument("-a", "--activation", default="Linear", help="Activation function of discriminiation network in RecDiscNet: Linear or Sigmoid")
parser.add_argument("-ar", "--activation_rec", default="Linear", help="Activationfunction of reconstruction network in RecDiscNet or other networks: Linear or Sigmoid")
parser.add_argument("-lr", "--learningrate", default=0.001, type=float, help="Learning rate")
parser.add_argument("-pw", "--positivweight", default=20, type=int, help="Positive weight for RecDiscNet loss. pw > 1 emphasized class 1.")
parser.add_argument("-lw", "--lossweight", default=5, type=int, help="Loss weight for RecDiscNet loss. lw > 1 emphasized discrimination.")
parser.add_argument("-leaky", "--leaky_relu", default="False", help="Use of leakyrelu. Otherwise a normal ReLu Unit is used.")
parser.add_argument("-an", "--anomaly", default="Mix", help="Choose anomaly: Iso, Normal1, Uniform1, Normal2, Uniform2, Mix. Whereby 1 is random and 2 directionally.")
parser.add_argument("-d", "--distribution", default="Full", help="Center of anomaly distribution is half the center of brain matter distribution if not set to Full.")

args = vars(parser.parse_args())
mode = args["mode"]
test_this_model = args["test_this_model"]
epochs = args["epochs"]
network = args["network"]
latent_dim = args["latent_dim"]
run = args["run"]
rec_filter = args["filter"]
ac_function = args["activation"]
ac_function_rec = args["activation_rec"]
anomaly_setting = args["anomaly"]
anomaly_distribution = args["distribution"]
learning_rate = args["learningrate"]
positiv_weight = args["positivweight"]
loss_weight = args["lossweight"]
leaky_relu = args["leaky_relu"]

# Batch size
if (network == "UNet") and run == 15: # for testing on smaller GPUs
  batch_size = 2
else:
  batch_size = 4

# Params
vanilla_params = {"LR": learning_rate,
  "weight_decay": 0.0,
  "scheduler_gamma": 0.95,
  "kld_weight": 0.0000122}      # (input.shape.flatten / latent space dimensions)^-1}

unet_params = {"LR": learning_rate,
  "weight_decay": 0.0,
  "scheduler_gamma": 0.95,
  "kld_weight": False}

rec_disc_params = {"LR": learning_rate,
  "weight_decay": 0.0,
  "scheduler_gamma": 0.95}

# Load data
img_path_uka = '/work/scratch/ecke/Masterarbeit/Data'
train = glob(opj(img_path_uka, "Train", "vp*"))
train.sort()
test = glob(opj(img_path_uka, "Test", "vp*"))
test.sort()
test_mask = glob(opj(img_path_uka, "Test", "mask*"))
test_mask.sort()
test_b0_brainmask = glob(opj(img_path_uka, "Test", "b0_brainmask*"))
test_b0_brainmask.sort()

# k-fold cross validation
split = 24 #used in dataset
if run == 1:
  uka_subjects = {"training": train[0:24], "validation": train[24:28], "test": test[0:32],
                  "test_mask": test[0:32]}
elif run == 2:
  uka_subjects = {"training": train[4:28], "validation": train[0:4], "test": test[0:32],
                  "test_mask": test[0:32]}
elif run == 3:
  uka_subjects = {"training": train[0:4]+train[8:28], "validation": train[4:8], "test": test[0:32],
                  "test_mask": test[0:32]}
elif run == 4:
  uka_subjects = {"training": train[0:8]+train[12:28], "validation": train[8:12], "test": test[0:32],
                  "test_mask": test[0:32]}
elif run == 5:
  uka_subjects = {"training": train[0:12]+train[16:28], "validation": train[12:16], "test": test[0:32],
                  "test_mask": test[0:32]}
elif run == 6:
  uka_subjects = {"training": train[0:16]+train[20:28], "validation": train[16:20], "test": test[0:32],
                  "test_mask": test[0:32]}
elif run == 7:
  uka_subjects = {"training": train[0:20]+train[24:28], "validation": train[20:24], "test": test[0:32],
                  "test_mask": test[0:32]}
else:
  random.shuffle(train)
  uka_subjects = {"training": train[0:split], "validation": train[split:28], "test": test[0:32],
                  "test_mask": test[0:32]}

# Names and paths
Tensor = TypeVar('torch.tensor')
log_dir = "/work/scratch/ecke/Masterarbeit/logs/Callback"
log_dir_logger = "/work/scratch/ecke/Masterarbeit/logs/Logger"
save_path = opj("/work/scratch/ecke/Masterarbeit/logs/Callback", test_this_model)
results_path = opj("/work/scratch/ecke/Masterarbeit/Results", test_this_model)
data_drop_off = opj("/work/scratch/ecke/Masterarbeit/logs/DataDropOff", test_this_model) 

# File names of results
anomaly_testing = True
if (network == "RecDisc") or (network == "RecDiscUnet"):
  if anomaly_testing == True:
      file_name = network + '-{epoch:02d}-{val_loss:.2f}' + '-r=' + str(run) + '-an=' + str(anomaly_setting) + '-d=' + str(anomaly_distribution)
  else:
    file_name = network + '-{epoch:02d}-{val_loss:.2f}' + '-r=' + str(run) + '-f=' + str(rec_filter) + '-ar=' + str(ac_function_rec) + '-a=' + str(ac_function) + '-leaky=' + str(leaky_relu)
elif (network == "VanillaVAE"):
  file_name = network + '-{epoch:02d}-{val_loss:.2f}' + '-r=' + str(run) + '-ldim=' + str(latent_dim) + '-ar=' + str(ac_function_rec)
elif (network == "UNet"):
  file_name = network + '-{epoch:02d}-{val_loss:.2f}' + '-r=' + str(run) + '-ar=' + str(ac_function_rec) + '-f=' + str(rec_filter)
else:
  file_name = network + '-{epoch:02d}-{val_loss:.2f}' + '-r=' + str(run)

# ID for direct testing after a run (train+test)
def create_id(test_id):
  global test_this_model
  test_this_model = test_id
  global save_path
  save_path = opj("/work/scratch/ecke/Masterarbeit/logs/Callback", test_this_model)
  global results_path
  results_path = opj("/work/scratch/ecke/Masterarbeit/Results", test_this_model)
  global data_drop_off
  data_drop_off = opj("/work/scratch/ecke/Masterarbeit/logs/DataDropOff", test_this_model)
  global mode
  mode = "test"
  return