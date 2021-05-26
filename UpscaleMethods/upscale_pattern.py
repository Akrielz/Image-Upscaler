from Utility.image_handler import *
from Utility.math_utility import *
import numpy as np


def analisePattern(image_name, rescaled_name, patch_factor=2):
    image_handler_original = ImageHandler()
    image_handler_original.read_image(image_name)

    image_handler_rescaled = ImageHandler()
    image_handler_rescaled.read_image(rescaled_name)

    width = image_handler_rescaled.width
    height = image_handler_rescaled.height

    pixels_original = image_handler_original.get_pixels_pointer()
    pixels_rescaled = image_handler_rescaled.get_pixels_pointer()

    pattern = []
    for x in range(patch_factor-1, width, patch_factor):
        row_pattern = []
        for y in range(patch_factor-1, height, patch_factor):

            local_pattern = []
            for xCur in range(x-patch_factor+1, x+1):
                for yCur in range(y-patch_factor+1, y+1):
                    originalPixelColors = np.asarray(pixels_original[xCur, yCur])
                    rescaledPixelColors = np.asarray(pixels_rescaled[xCur, yCur])
                    for i in range(len(rescaledPixelColors)):
                        if rescaledPixelColors[i] < EPS:
                            rescaledPixelColors[i] = 1

                    pixel_transform = originalPixelColors / rescaledPixelColors
                    local_pattern.append(pixel_transform)

            row_pattern.append(local_pattern)

        pattern.append(row_pattern)
    return pattern


def applyPatternCorrection(image_name, pattern):
    image_handler = ImageHandler()
    image_handler.read_image(image_name)

    width = image_handler.width
    height = image_handler.height

    pixels = image_handler.get_pixels_pointer()

    pattern_len_x2 = int(len(pattern[0][0]))
    pattern_len = int(pattern_len_x2**1/2)

    pattern_x = 0
    for x in range(pattern_len_x2-1, width, pattern_len_x2):
        pattern_y = 0
        for y in range(pattern_len_x2 - 1, height, pattern_len_x2):

            pattern_k = 0
            for xPattern in range(x-pattern_len_x2+1, x+1, pattern_len):
                for yPattern in range(y-pattern_len_x2+1, y+1, pattern_len):

                    for xCur in range(xPattern, xPattern+pattern_len):
                        for yCur in range(yPattern, yPattern+pattern_len):
                            pixelColors = np.asarray(pixels[xCur, yCur])
                            pixelColors = pixelColors * pattern[pattern_x][pattern_y][pattern_k]
                            pixelColors = (int(pixelColors[0]), int(pixelColors[1]), int(pixelColors[2]))
                            pixels[x, y] = pixelColors
                            pass

                    pattern_k += 1
                    pass

            pattern_y += 1

        pattern_x += 1


class PatternRecognizer:
    def __init__(self):
        self.image_handler_initial = None
        self.image_handler_dowscaled = None
        self.image_handler_rescaled = None
        self.image_handler_upscaled = None

    def upscale_image(self, image_name, save_name, patch_factor=2):
        downscaled_x2_name = "./Temporal/temp_downscaled_d2.png"
        rescaled_name = "./Temporal/temp_rescaled.png"

        self.image_handler_initial = ImageHandler()
        self.image_handler_initial.read_image(image_name)
        self.image_handler_initial.rescale_image_by_factor(scale_factor=1.0/patch_factor, new_image_name=downscaled_x2_name)

        self.image_handler_dowscaled = ImageHandler()
        self.image_handler_dowscaled.read_image(downscaled_x2_name)
        self.image_handler_dowscaled.rescale_image_by_factor(scale_factor=patch_factor, new_image_name=rescaled_name)

        pattern = analisePattern(image_name, rescaled_name, patch_factor)

        self.image_handler_initial.rescale_image_by_factor(scale_factor=patch_factor, new_image_name=save_name)

        applyPatternCorrection(save_name, pattern)
