# noinspection PyUnresolvedReferences
from keras.models import Model, load_model

# noinspection PyUnresolvedReferences
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten

# noinspection PyUnresolvedReferences
from keras.optimizers import Adam, RMSprop

# noinspection PyUnresolvedReferences
from keras import backend as K

# noinspection PyUnresolvedReferences
import keras

from Utility.image_handler import *
from Utility.os_utility import *
from Utility.constants import *

import numpy as np
import threading

import math


class ModelNNPixelX2:
    def __init__(self, lr=1e-4, nr_channels=3):
        self.nr_input = 4
        self.nr_hidden = [256, 256, 256]
        self.nr_output = 4
        self.nr_color_channels = nr_channels

        self.input_shape = (self.nr_input, self.nr_color_channels)

        self.layer_input = Input(shape=self.input_shape)

        self.layers_hidden = []

        layer_last = self.layer_input
        for nr_units in self.nr_hidden:
            layer = Dense(nr_units, activation="relu", kernel_initializer='he_uniform')(layer_last)

            layer_last = layer
            self.layers_hidden.append(layer)

        self.layer_output = Dense(nr_channels, activation="relu", kernel_initializer='he_uniform')(
            self.layers_hidden[-1])

        self.model = Model(inputs=self.layer_input, outputs=self.layer_output)
        # self.model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=lr))
        self.model.compile(loss=keras.losses.MeanSquaredError(), optimizer=Adam(lr=lr))


class ProgressiveNN:
    def __init__(self, nr_channels=3):
        self.lr = 1e-4
        self.nr_channels = nr_channels

        self.neural_network_x2 = ModelNNPixelX2(lr=self.lr, nr_channels=nr_channels).model
        self.neural_network_x4 = ModelNNPixelX2(lr=self.lr, nr_channels=nr_channels).model
        self.neural_network_x8 = ModelNNPixelX2(lr=self.lr, nr_channels=nr_channels).model

        self.image_handler_initial = None
        self.image_handler_downscaled = None
        self.image_handler_rescaled = None
        self.image_handler_upscaled = None

        self.agent_id = None
        with open(nn_progressive_save_path + "/index.in", "r") as file:
            self.agent_id = int(file.readline())

    def train(self, training_directory="./data/original", method="x2", start=0):
        files, directories = get_folder_content(training_directory, everything=True)
        n = len(files)

        neural_network = {
            "x2": self.neural_network_x2,
            "x4": self.neural_network_x4,
            "x8": self.neural_network_x8,
        }

        get_training_data = {
            "x2": self.__get_training_data_x2,
            "x4": self.__get_training_data_x4,
            "x8": self.__get_training_data_x8,
        }

        for i, file in enumerate(files):
            if i < start:
                continue

            print(f"File[{i}] / {n}")
            X_mini, Y_mini = get_training_data[method](file)
            X_mini = [np.array(X_mini) / 255]
            Y_mini = [np.array(Y_mini) / 255]
            neural_network[method].fit(X_mini, Y_mini, verbose=1)
            self.save()

        pass

    def upscale_x2(self, image_name, save_name, color_approx=False, factor=1):
        self.__predict_upscale(image_name, save_name, image_name,
                               self.neural_network_x2, color_approx, factor, upscale=2)

    def upscale_x4(self, image_name, save_name, color_approx=False, factor=1):
        upscale_x2_name = "./Temporal/temp_upscaled_u2.png"
        self.upscale_x2(image_name, upscale_x2_name, color_approx=color_approx, factor=1)
        self.__predict_upscale(upscale_x2_name, save_name, image_name,
                               self.neural_network_x4, color_approx, factor, upscale=4)

    def upscale_x8(self, image_name, save_name, color_approx=False, factor=1):
        upscale_x4_name = "./Temporal/temp_upscaled_u4.png"
        self.upscale_x4(image_name, upscale_x4_name, color_approx=color_approx, factor=1)
        self.__predict_upscale(upscale_x4_name, save_name, image_name,
                               self.neural_network_x8, True, factor, upscale=8)

    def __predict_upscale(self, image_name, save_name, original_image, upscale_nn=None, color_approx=False, factor=1, upscale=2):
        # Cheap Upscale x1 -> x2
        self.image_handler_initial = ImageHandler()
        self.image_handler_initial.read_image(image_name)

        upscaled_width = self.image_handler_initial.width * 2
        upscaled_height = self.image_handler_initial.height * 2

        self.image_handler_initial.rescale_image(upscaled_width, upscaled_height, save_name)

        # Start from the cheap upscale
        self.image_handler_upscaled = ImageHandler()
        self.image_handler_upscaled.read_image(save_name)

        pixels = self.image_handler_upscaled.get_pixels_pointer()

        # Get NN version
        X = self.__get_image_patches_x2(pixels, upscaled_width, upscaled_height)
        X = [np.array(X) / 255]
        Y = upscale_nn.predict(X)

        # Normalize the output
        max_val = 0

        for patch in Y:
            for pixel in patch:
                for channel in pixel:
                    max_val = max(max_val, channel)

        for i, patch in enumerate(Y):
            for j, pixel in enumerate(patch):
                for k, channel in enumerate(pixel):
                    Y[i][j][k] /= max_val
                    pass

        Y = np.array(Y) * 255

        # Paint the output
        self.image_handler_initial = ImageHandler()
        self.image_handler_initial.read_image(original_image)

        self.__set_image_pixels_x2(pixels, upscaled_width, upscaled_height, Y,
                                   self.image_handler_initial.pixels, color_approx,
                                   factor, upscale=upscale)
        self.image_handler_upscaled.save_image(save_name)
        pass

    def __set_image_pixels_x2(self, pixels, width, height, Y, original_pixels, color_approx, factor, upscale=2):
        index = 0
        for x_patch in range(width // 2):
            for y_patch in range(height // 2):
                x = x_patch
                y = y_patch

                pixels_positions = [(2 * x, 2 * y), (2 * x + 1, 2 * y),
                                    (2 * x, 2 * y + 1), (2 * x + 1, 2 * y + 1)]

                local_pixels = Y[index]
                original_pixel = original_pixels[x // (upscale / 2), y // (upscale / 2)]

                if color_approx:
                    pixels_mean = [0, 0, 0]
                    for future_pixel in local_pixels:
                        pixels_mean[0] += future_pixel[0]
                        pixels_mean[1] += future_pixel[1]
                        pixels_mean[2] += future_pixel[2]

                    adjust = [0, 0, 0]
                    for i in range(len(pixels_mean)):
                        pixels_mean[i] /= 4
                        adjust[i] = original_pixel[i] / (pixels_mean[i] + EPS)

                    for future_pixel in local_pixels:
                        for i, adjusted_channel in enumerate(adjust):
                            future_pixel[i] *= adjusted_channel
                            dist = future_pixel[i] - original_pixel[i]
                            future_pixel[i] = original_pixel[i] + dist * factor

                j = 0
                for x, y in pixels_positions:
                    # print(local_pixels[j])
                    pixels[x, y] = convert_pixel_to(local_pixels[j], nr_channels=self.nr_channels)
                    j += 1

                index += 1

    def __get_image_patches_line_x2(self, pixels, height, line, pixels_in_patch_line):
        for y_patch in range(height // 2):
            x = line
            y = y_patch

            pixels_positions = [(2 * x, 2 * y), (2 * x + 1, 2 * y),
                                (2 * x, 2 * y + 1), (2 * x + 1, 2 * y + 1)]

            patch_pixels = np.array([convert_pixel_to(pixels[x, y], self.nr_channels)
                                     for (x, y) in pixels_positions])
            pixels_in_patch_line.append(patch_pixels)

    def __get_image_patches_x2(self, pixels, width, height):
        pixels_in_patch = [[] for i in range(width // 2)]

        threads = []
        for x_patch in range(width // 2):
            threads.append(threading.Thread(target=self.__get_image_patches_line_x2,
                                            args=(pixels, height, x_patch, pixels_in_patch[x_patch])))

        for i in range(width // 2):
            threads[i].start()

        for i in range(width // 2):
            threads[i].join()

        all_pixels = []
        for i in range(width // 2):
            all_pixels.extend(pixels_in_patch[i])

        return all_pixels

    def __get_training_data_x2(self, image_name):
        # Downscale x2
        self.image_handler_initial = ImageHandler()
        self.image_handler_initial.read_image(image_name)

        downscaled_width = self.image_handler_initial.width // 2
        downscaled_height = self.image_handler_initial.height // 2

        self.image_handler_initial.rescale_image(downscaled_width, downscaled_height, "./Temporal/temp_downscaled.png")

        # Cheap Rescale x2
        self.image_handler_downscaled = ImageHandler()
        self.image_handler_downscaled.read_image("./Temporal/temp_downscaled.png")

        rescaled_width = downscaled_width * 2
        rescaled_height = downscaled_height * 2

        self.image_handler_downscaled.rescale_image(rescaled_width, rescaled_height, "./Temporal/temp_rescaled.png")

        # Get Training Data
        self.image_handler_rescaled = ImageHandler()
        self.image_handler_rescaled.read_image("./Temporal/temp_rescaled.png")

        self.image_handler_initial = ImageHandler()
        self.image_handler_initial.read_image(image_name)

        initial_pixels = self.image_handler_initial.get_pixels_pointer()
        rescaled_pixels = self.image_handler_rescaled.get_pixels_pointer()

        X_train = self.__get_image_patches_x2(rescaled_pixels, rescaled_width, rescaled_height)
        Y_train = self.__get_image_patches_x2(initial_pixels, rescaled_width, rescaled_height)

        return X_train, Y_train

    def __get_training_data_x4(self, image_name):
        downscale_x4_name = "./Temporal/temp_downscaled_d4.png"
        rescaled_x2_name = "./Temporal/temp_downscaled_d2.png"
        rescaled_x1_name = "./Temporal/temp_rescaled.png"

        # Downscale x4
        self.image_handler_initial = ImageHandler()
        self.image_handler_initial.read_image(image_name)

        downscaled_width = self.image_handler_initial.width // 4
        downscaled_height = self.image_handler_initial.height // 4

        self.image_handler_initial.rescale_image(downscaled_width, downscaled_height, downscale_x4_name)

        # Best Rescale x2 (downscaled x2)
        self.upscale_x2(image_name=downscale_x4_name, save_name=rescaled_x2_name, color_approx=True)

        # Cheap Rescale (initial)
        self.image_handler_downscaled = ImageHandler()
        self.image_handler_downscaled.read_image(rescaled_x2_name)

        rescaled_width = downscaled_width * 4
        rescaled_height = downscaled_height * 4

        self.image_handler_downscaled.rescale_image(rescaled_width, rescaled_height, rescaled_x1_name)

        # Get Training Data
        self.image_handler_rescaled = ImageHandler()
        self.image_handler_rescaled.read_image(rescaled_x1_name)

        self.image_handler_initial = ImageHandler()
        self.image_handler_initial.read_image(image_name)
        initial_pixels = self.image_handler_initial.get_pixels_pointer()
        rescaled_pixels = self.image_handler_rescaled.get_pixels_pointer()

        X_train = self.__get_image_patches_x2(rescaled_pixels, rescaled_width, rescaled_height)
        Y_train = self.__get_image_patches_x2(initial_pixels, rescaled_width, rescaled_height)

        return X_train, Y_train

    def __get_training_data_x8(self, image_name):
        downscale_x8_name = "./Temporal/temp_downscaled_d8.png"
        rescaled_x2_name = "./Temporal/temp_downscaled_d2.png"
        rescaled_x1_name = "./Temporal/temp_rescaled.png"

        # Downscale x1 -> x8
        self.image_handler_initial = ImageHandler()
        self.image_handler_initial.read_image(image_name)

        downscaled_width = self.image_handler_initial.width // 8
        downscaled_height = self.image_handler_initial.height // 8

        self.image_handler_initial.rescale_image(downscaled_width, downscaled_height, downscale_x8_name)

        # Best Rescale x8 -> x2 (downscaled x2)
        self.upscale_x4(image_name=downscale_x8_name, save_name=rescaled_x2_name, color_approx=True)

        # Cheap Rescale (initial)
        self.image_handler_downscaled = ImageHandler()
        self.image_handler_downscaled.read_image(rescaled_x2_name)

        rescaled_width = downscaled_width * 8
        rescaled_height = downscaled_height * 8

        self.image_handler_downscaled.rescale_image(rescaled_width, rescaled_height, rescaled_x1_name)

        # Get Training Data
        self.image_handler_rescaled = ImageHandler()
        self.image_handler_rescaled.read_image(rescaled_x1_name)

        self.image_handler_initial = ImageHandler()
        self.image_handler_initial.read_image(image_name)
        initial_pixels = self.image_handler_initial.get_pixels_pointer()
        rescaled_pixels = self.image_handler_rescaled.get_pixels_pointer()

        X_train = self.__get_image_patches_x2(rescaled_pixels, rescaled_width, rescaled_height)
        Y_train = self.__get_image_patches_x2(initial_pixels, rescaled_width, rescaled_height)

        return X_train, Y_train

    def load(self, agent_id: int, to_compile=False):
        self.neural_network_x2 = load_model(nn_progressive_save_path + "/NNP_x2_" + str(agent_id) + ".h5",
                                            compile=to_compile)
        self.neural_network_x4 = load_model(nn_progressive_save_path + "/NNP_x4_" + str(agent_id) + ".h5",
                                            compile=to_compile)
        self.neural_network_x8 = load_model(nn_progressive_save_path + "/NNP_x8_" + str(agent_id) + ".h5",
                                            compile=to_compile)

    def save(self):
        self.neural_network_x2.save(nn_progressive_save_path + "/NNP_x2_" + str(self.agent_id) + ".h5")
        self.neural_network_x4.save(nn_progressive_save_path + "/NNP_x4_" + str(self.agent_id) + ".h5")
        self.neural_network_x8.save(nn_progressive_save_path + "/NNP_x8_" + str(self.agent_id) + ".h5")

        self.agent_id += 1
        with open(nn_progressive_save_path + "/index.in", "w") as file:
            file.write(str(self.agent_id))
