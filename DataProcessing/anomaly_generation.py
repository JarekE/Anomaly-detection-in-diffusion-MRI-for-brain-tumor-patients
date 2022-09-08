import random
import matplotlib.pyplot as plt
from config import anomaly_setting
import raster_geometry as rg
import numpy as np


def anomaly_generation(input_data, mask):
    input_anomaly = input_data.copy()
    block = np.zeros_like(input_data)
    z_print = 30
    x, y, z = 32, 40, 32

    # Number of tumors
    random_value = random.uniform(0, 1)
    if random_value > 0.80:
        number_tumor = 2
    elif random_value < 0.60:
        number_tumor = 1
    else:
        number_tumor = 0

    for tumor in range(number_tumor):
        # Place anomaly
        x_sign = [-1, 1][random.randrange(2)]
        y_sign = [-1, 1][random.randrange(2)]
        z_sign = [-1, 1][random.randrange(2)]

        # Size and shape of anomaly
        random_size = random.randint(7, 11)
        #anomaly_block = rg.sphere(2 * random_size, random_size).astype(int)
        anomaly_block = rg.ellipsoid(2 * random_size, (random_size * random.uniform(0.5, 1), random_size * random.uniform(0.5, 1), random_size * random.uniform(0.5, 1)))
        channels = [anomaly_block for _ in range(64)]
        anomaly_block = np.stack(channels, axis=0)

        high_value = 0.3

        if anomaly_setting == "Iso":
            # Isotropic
            anomaly_block = np.multiply(anomaly_block, random.uniform(0.1, high_value))
        elif anomaly_setting == "Gauss1":
            # Random noise with centre value between 0.1 and 0.3
            gaussian_noise = np.random.normal(random.uniform(0.1, high_value), 0.1, size=anomaly_block.shape)
            anomaly_block = np.multiply(anomaly_block, gaussian_noise)
        elif anomaly_setting == "Gauss2":
            # Random vector in channel dimension asigned to each voxel of tumor (Equal for all voxel per channel)
            random_vector = (np.random.rand(64) * high_value)
            for i in range(3):
                channels = [random_vector for _ in range(2*random_size)]
                random_vector = np.stack(channels, axis=1)
            anomaly_block = np.multiply(anomaly_block, random_vector)
        else:
            random_value2 = random.uniform(0, 1)
            if random_value2 > 0.66:
                anomaly_block = np.multiply(anomaly_block, random.uniform(0.1, high_value))
            elif random_value2 < 0.33:
                gaussian_noise = np.random.normal(random.uniform(0.1, high_value), 0.1, size=anomaly_block.shape)
                anomaly_block = np.multiply(anomaly_block, gaussian_noise)
            else:
                random_vector = (np.random.rand(64) * high_value)
                for i in range(3):
                    channels = [random_vector for _ in range(2 * random_size)]
                    random_vector = np.stack(channels, axis=1)
                anomaly_block = np.multiply(anomaly_block, random_vector)

        # Place block in input-shaped array
        random_x = x_sign * random.randint(4, 20) + x
        random_y = y_sign * random.randint(4, 20) + y
        random_z = z_sign * random.randint(4, 20) + z

        while not (mask[random_x, random_y, random_z].item() == 1.0 and mask[
            random_x + 1, random_y + 1, random_z + 1].item() == 1.0):
            random_x = x_sign * random.randint(4, 20) + x
            random_y = y_sign * random.randint(4, 20) + y
            random_z = z_sign * random.randint(4, 20) + z

        # Use to print
        if tumor == 0:
            z_print = random_z

        block[:, random_x - random_size:random_x + random_size, random_y - random_size:random_y + random_size,
        random_z - random_size:random_z + random_size] = anomaly_block

        # Make space for the block (zero all elements)
        input_anomaly[...] = np.multiply(input_anomaly[...], np.where(block[...] > 0, 0, 1))
        input_anomaly[...] = np.add(input_anomaly[...], block[...])

    # Reconstruction map
    reconstructive_map = np.where(block[0, :, :, :] != 0, 1, 0)
    reconstructive_map = reconstructive_map[None, :, :, :]
    #reconstructive_map_background = torch.where(block[:, 0, :, :, :] == 0, 1, 0)
    #reconstructive_map = torch.cat((reconstructive_map_tumor[:, None, :, :, :], reconstructive_map_background[:, None, :, :, :]), dim=1)

    return input_anomaly, np.float32(reconstructive_map), z_print