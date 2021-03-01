# noinspection PyUnresolvedReferences
from keras.models import Model, load_model

# noinspection PyUnresolvedReferences
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten

# noinspection PyUnresolvedReferences
from keras.optimizers import Adam, RMSprop

# noinspection PyUnresolvedReferences
from keras import backend as K

from Utility.image_handler import *
from Utility.os_utility import *
from Utility.constants import *

import numpy as np


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
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=lr))


class ProgressiveNN:
    def __init__(self, nr_channels=3):
        self.lr = 1e-4
        self.nr_channels = nr_channels

        self.neural_network_x2 = ModelNNPixelX2(lr=self.lr, nr_channels=nr_channels).model

        self.image_handler_initial = None
        self.image_handler_downscaled = None
        self.image_handler_rescaled = None
        self.image_handler_upscaled = None

        self.agent_id = None
        with open(nn_progressive_save_path + "/index.in", "r") as file:
            self.agent_id = int(file.readline())

    def train_x2(self, training_directory="./data/original"):
        files, directories = get_folder_content(training_directory)
        X, Y = [], []
        for file in files:
            X_mini, Y_mini = self.get_training_data_x2(training_directory + "/" + file)
            X.extend(X_mini)
            Y.extend(Y_mini)

        X = [np.array(X) / 255]
        Y = [np.array(Y) / 255]

        self.neural_network_x2.fit(X, Y, verbose=1)
        pass

    def upscale_x2(self, image_name, save_name):
        # Cheap Upscale x2
        self.image_handler_initial = ImageHandler()
        self.image_handler_initial.read_image(image_name)

        upscaled_width = self.image_handler_initial.width * 2
        upscaled_height = self.image_handler_initial.height * 2

        self.image_handler_initial.rescale_image(upscaled_width, upscaled_height, save_name)

        self.image_handler_upscaled = ImageHandler()
        self.image_handler_upscaled.read_image(save_name)

        pixels = self.image_handler_upscaled.get_pixels_pointer()

        X = self.get_image_patches_x2(pixels, upscaled_width, upscaled_height)
        X = [np.array(X) / 255]
        Y = self.neural_network_x2.predict(X)

        max_val = 0
        for patch in Y:
            for pixel in patch:
                for channel in pixel:
                    max_val = max(max_val, channel)

        for i, patch in enumerate(Y):
            for j, pixel in enumerate(patch):
                for k, channel in enumerate(pixel):
                    Y[i][j][k] /= max_val

        Y = np.array(Y) * 255

        self.set_image_pixels_x2(pixels, upscaled_width, upscaled_height, Y)
        # self.image_handler_upscaled.save_image(save_name+"2.png")
        self.image_handler_upscaled.save_image(save_name)

    def set_image_pixels_x2(self, pixels, width, height, Y):
        index = 0
        for x_patch in range(width // 2):
            for y_patch in range(height // 2):
                x = x_patch
                y = y_patch

                pixels_positions = [(2 * x, 2 * y), (2 * x + 1, 2 * y),
                                    (2 * x, 2 * y + 1), (2 * x + 1, 2 * y + 1)]

                local_pixels = Y[index]

                j = 0
                for x, y in pixels_positions:
                    # print(local_pixels[j])
                    pixels[x, y] = convert_pixel_to(local_pixels[j], nr_channels=self.nr_channels)
                    j += 1

                index += 1

    def get_image_patches_x2(self, pixels, width, height):
        pixels_in_patch = []

        for x_patch in range(width // 2):
            for y_patch in range(height // 2):
                x = x_patch
                y = y_patch

                pixels_positions = [(2 * x, 2 * y), (2 * x + 1, 2 * y),
                                    (2 * x, 2 * y + 1), (2 * x + 1, 2 * y + 1)]

                patch_pixels = np.array([convert_pixel_to(pixels[x, y], self.nr_channels)
                                         for (x, y) in pixels_positions])
                pixels_in_patch.append(patch_pixels)

        return pixels_in_patch

    def get_training_data_x2(self, image_name):
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

        initial_pixels = self.image_handler_initial.get_pixels_pointer()
        rescaled_pixels = self.image_handler_rescaled.get_pixels_pointer()

        X_train = self.get_image_patches_x2(rescaled_pixels, rescaled_width, rescaled_height)
        Y_train = self.get_image_patches_x2(initial_pixels, rescaled_width, rescaled_height)

        return X_train, Y_train

    def load(self, agent_id: int, to_compile=False):
        self.neural_network_x2 = load_model(nn_progressive_save_path + "/NNP_x2_" + str(agent_id) + ".h5",
                                            compile=to_compile)

    def save(self):
        self.neural_network_x2.save(nn_progressive_save_path + "/NNP_x2_" + str(self.agent_id) + ".h5")
        self.agent_id += 1
        with open(nn_progressive_save_path + "/index.in", "w") as file:
            file.write(str(self.agent_id))
