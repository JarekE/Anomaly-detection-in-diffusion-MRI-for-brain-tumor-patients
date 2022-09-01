import numpy as np
import random
import matplotlib.pyplot as plt
import config
import raster_geometry as rg
import torch


def anomaly_generation(input, mask):

    # Generate anomaly block
    input_anomaly = input
    sign = [-1, 1][random.randrange(2)]
    x, y, z = 32, 40, 32
    random_size = random.randint(5, 10)
    anomaly_block = torch.from_numpy(rg.sphere(2*random_size, random_size).astype(int))

    # Isotropic
    if random.uniform(0, 1) > 0.65:
        anomaly_block = torch.mul(anomaly_block, random.uniform(0, 1))
    # Random noise with centre value between 0.25 and 0.55
    else:
        gaussian_noise = np.random.normal(random.uniform(0.25, 0.55), 0.1, size=anomaly_block.shape)
        anomaly_block = torch.mul(anomaly_block, torch.from_numpy(gaussian_noise))

    # Place block in input-shaped array
    block = torch.zeros_like(input)
    random_x = sign * random.randint(4, 20) + x
    random_y = sign * random.randint(4, 20) + y
    random_z = sign * random.randint(4, 20) + z
    block[:, :, random_x - random_size:random_x + random_size, random_y - random_size:random_y + random_size,
    random_z - random_size:random_z + random_size] = anomaly_block

    # Check if point of approximate centre is inside the brainmask (should be improved)
    while not torch.equal(mask[:, random_x, random_y, random_z].cpu(), torch.ones(4)):
        random_x = sign * random.randint(4, 20) + x
        random_y = sign * random.randint(4, 20) + y
        random_z = sign * random.randint(4, 20) + z

        block = torch.zeros_like(input)
        block[:, :, random_x - random_size:random_x + random_size, random_y - random_size:random_y + random_size,
        random_z - random_size:random_z + random_size] = anomaly_block


    # Make space for the block (zero all elements)
    input_anomaly = torch.mul(input_anomaly, torch.where(block > 0, 0, 1))

    # Add block to input, not for testdata
    if config.mode == "test":
        input = input
    else:
        input = torch.add(input_anomaly, block)

    # Reconstruction map
    reconstructive_map = torch.where(block[:, 0, :, :, :] != 0, 1, 0)
    reconstructive_map = reconstructive_map[:, None, :, :, :]
    #reconstructive_map_background = torch.where(block[:, 0, :, :, :] == 0, 1, 0)
    #reconstructive_map = torch.cat((reconstructive_map_tumor[:, None, :, :, :], reconstructive_map_background[:, None, :, :, :]), dim=1)

    return input, reconstructive_map, random_z