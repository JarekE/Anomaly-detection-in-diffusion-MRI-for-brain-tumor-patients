import random
from config import anomaly_setting, anomaly_distribution
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
        random_size = random.randint(5, 11)
        anomaly_block = rg.ellipsoid(2 * random_size, (random_size * random.uniform(0.5, 1), random_size * random.uniform(0.5, 1), random_size * random.uniform(0.5, 1)))
        channels = [anomaly_block for _ in range(64)]
        anomaly_block = np.stack(channels, axis=0)

        # Define values for anomalies
        mean = 0.4
        if anomaly_distribution != "Full":
            mean = 0.2
        sd = 0.12
        min = mean - sd
        max = mean + sd

        if anomaly_setting == "Iso":
            # Isotropic
            anomaly_block = np.multiply(anomaly_block, random.uniform(min, max))
        elif anomaly_setting == "Uniform1":
            # Random noise with centre value between 0.1 and 0.3
            noise = np.random.uniform(min, max, size=anomaly_block.shape)
            anomaly_block = np.multiply(anomaly_block, noise)
        elif anomaly_setting == "Normal1":
            # Random noise with centre value between 0.1 and 0.3
            gaussian_noise = np.random.normal(mean, sd, size=anomaly_block.shape)
            anomaly_block = np.multiply(anomaly_block, gaussian_noise)
        elif anomaly_setting == "Uniform2":
            # Random vector in channel dimension asigned to each voxel of tumor (Equal for all voxel per channel)
            random_vector = np.random.uniform(min, max, size=64)
            for i in range(3):
                channels = [random_vector for _ in range(2*random_size)]
                random_vector = np.stack(channels, axis=1)
            anomaly_block = np.multiply(anomaly_block, random_vector)
        elif anomaly_setting == "Normal2":
            # Random vector in channel dimension asigned to each voxel of tumor (Equal for all voxel per channel)
            random_vector = np.random.normal(mean, sd, size=64)
            for i in range(3):
                channels = [random_vector for _ in range(2*random_size)]
                random_vector = np.stack(channels, axis=1)
            anomaly_block = np.multiply(anomaly_block, random_vector)
        else:
            random_value2 = random.uniform(0, 1)
            random_value3 = random.uniform(0, 1)
            if random_value2 > 0.66:
                anomaly_block = np.multiply(anomaly_block, random.uniform(min, max))
            elif random_value2 < 0.33:
                if random_value3 > 0.5:
                    noise = np.random.uniform(min, max, size=anomaly_block.shape)
                    anomaly_block = np.multiply(anomaly_block, noise)
                else:
                    gaussian_noise = np.random.normal(mean, sd, size=anomaly_block.shape)
                    anomaly_block = np.multiply(anomaly_block, gaussian_noise)
            else:
                if random_value3 > 0.5:
                    random_vector = (np.random.uniform(min, max, size=64))
                    for i in range(3):
                        channels = [random_vector for _ in range(2 * random_size)]
                        random_vector = np.stack(channels, axis=1)
                    anomaly_block = np.multiply(anomaly_block, random_vector)
                else:
                    random_vector = (np.random.normal(mean, sd, size=64))
                    for i in range(3):
                        channels = [random_vector for _ in range(2 * random_size)]
                        random_vector = np.stack(channels, axis=1)
                    anomaly_block = np.multiply(anomaly_block, random_vector)

        # Place block in input-shaped array
        random_x = x_sign * random.randint(4, 20) + x
        random_y = y_sign * random.randint(4, 20) + y
        random_z = z_sign * random.randint(4, 20) + z

        # Make sure centre points of block are inside the brain matter
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

    return input_anomaly, np.float32(reconstructive_map), z_print