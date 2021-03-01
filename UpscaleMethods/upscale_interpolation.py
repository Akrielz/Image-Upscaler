from Utility.image_handler import *
from Utility.math_utility import *
import numpy as np


class Interpolator:
    def __init__(self):
        self.image_handler_initial = None
        self.image_handler_naive = None

    def upscale_image(self, image_name, distance, save_name):
        self.image_handler_initial = ImageHandler()
        self.image_handler_initial.read_image(image_name)

        new_width = self.image_handler_initial.width * distance - (distance-1)
        new_height = self.image_handler_initial.height * distance - (distance-1)

        self.image_handler_naive = ImageHandler()
        self.image_handler_initial.rescale_image(new_width, new_height, "./Temporal/temp.png")
        self.image_handler_naive.read_image("./Temporal/temp.png")

        pixels = self.image_handler_naive.get_pixels_pointer()

        for x in range(new_width):
            for y in range(new_height):
                if x % distance == 0 and y % distance == 0:
                    continue

                patch_x = (x-1) // distance
                if x == 0:
                    patch_x = 0

                patch_y = (y-1) // distance
                if y == 0:
                    patch_y = 0

                top = patch_y * distance
                left = patch_x * distance
                bot = top + distance
                right = left + distance

                corner_pixels = [pixels[left, top], pixels[right, top], pixels[left, bot], pixels[right, bot]]
                corner_positions = [(left, top), (right, top), (left, bot), (right, bot)]

                corner_distance = [distance_euclid_pos(pos, (x, y)) for pos in corner_positions]
                corner_distance = [dist for dist in corner_distance]
                max_distance = distance_euclid_pos(corner_positions[0], corner_positions[-1])

                corner_distance_inverse = [max_distance - dist for dist in corner_distance]
                inverse_distance_sum = sum(corner_distance_inverse)
                corner_percent = [dist / inverse_distance_sum for dist in corner_distance_inverse]

                max_percent_index = np.argmax(corner_percent)
                to_add = 0
                donate = 0.2
                for i, p in enumerate(corner_percent):
                    if i == max_percent_index:
                        continue
                    to_add += corner_percent[i] * donate
                    corner_percent[i] -= corner_percent[i] * donate
                corner_percent[max_percent_index] += to_add

                new_pixel = [0, 0, 0, 0]
                for i, pixel in enumerate(corner_pixels):

                    for j, pixel_value in enumerate(pixel):
                        pixel_true_value = pixel_value**2
                        new_pixel[j] += corner_percent[i]*pixel_true_value

                new_pixel = [int(value**(1/2)) for value in new_pixel]
                new_pixel = (new_pixel[0], new_pixel[1], new_pixel[2], new_pixel[3])

                pixels[x, y] = new_pixel

                # if x == 251 and y == 251:
                #     print(x, y, corner_positions)
                #
                #     pixels[x, y] = (0, 0, 0)
                #     for c in corner_positions:
                #         pixels[c[0], c[1]] = (0, 0, 0)

        self.image_handler_naive.save_image(save_name)
