# noinspection PyUnresolvedReferences
from PIL import Image, ImageEnhance

import numpy as np
from scipy import signal

from Utility.constants import *
from Utility.math_utility import *


def convert_pixel_to(pixel, nr_channels=3):
    new_pixel = [0, 0, 0, 0]

    for j, pixel_value in enumerate(pixel):
        new_pixel[j] = pixel_value

    if nr_channels == 3:
        new_pixel = (new_pixel[0], new_pixel[1], new_pixel[2])
    else:
        new_pixel = (new_pixel[0], new_pixel[1], new_pixel[2], new_pixel[3])
    return new_pixel


def get_common_part(image_name1, image_name2):
    im_handler1 = ImageHandler()
    im_handler1.read_image(image_name1)

    im_handler2 = ImageHandler()
    im_handler2.read_image(image_name2)

    minWidth = min(im_handler1.width, im_handler2.width)
    minHeight = min(im_handler1.height, im_handler2.height)

    pixels_img1 = im_handler1.get_pixels_copy(w=minWidth, h=minHeight)
    pixels_img2 = im_handler2.get_pixels_copy(w=minWidth, h=minHeight)

    pixels_img1 = np.asarray(pixels_img1)
    pixels_img2 = np.asarray(pixels_img2)

    return pixels_img1, pixels_img2


def calculate_psnr(image_name1, image_name2):
    pixels_img1, pixels_img2 = get_common_part(image_name1, image_name2)

    mse = np.mean((pixels_img1 - pixels_img2)**2)
    if abs(mse) < EPS:
        return float('inf')

    return 20*np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(image_name1, image_name2, K1=0.01, K2=0.03, L=255):
    img1, img2 = get_common_part(image_name1, image_name2)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    avg1 = np.average(img1)
    avg2 = np.average(img2)

    cov = covariance(img1, img2)
    var1 = covariance(img1, img1)
    var2 = covariance(img2, img2)

    return (2*avg1*avg2 + C1)*(2*cov + C2) / ((avg1**2 + avg2**2 + C1)*(var1 + var2 + C2))


class ImageHandler:
    def __init__(self):
        self.image = None
        self.pixels = None
        self.height = None
        self.width = None

    def read_image(self, image_name):
        self.image = Image.open(image_name)
        self.pixels = self.image.load()
        self.width, self.height = self.image.size
        return self.image

    def get_pixels_pointer(self):
        return self.pixels

    def get_pixels_copy(self, w=None, h=None):
        if w is None:
            w = self.width

        if h is None:
            h = self.height

        width, height = w, h
        pixels = []
        for x in range(width):
            row = []
            for y in range(height):
                row.append(self.pixels[x, y])

            pixels.append(row)
        return pixels

    def get_size(self):
        return self.image.size

    def save_image(self, image_name):
        self.image.save(image_name)

    def rescale_image_keep_ratio_width(self, new_width, new_image_name):
        width_percent = new_width / self.width
        new_height = int(self.height * width_percent)
        self.rescale_image(new_width, new_height, new_image_name)

    def rescale_image_keep_ratio_height(self, new_height, new_image_name):
        height_percent = new_height / self.height
        new_width = int(self.width * height_percent)
        self.rescale_image(new_width, new_height, new_image_name)

    def rescale_image(self, new_width, new_height, new_image_name):
        new_width = int(new_width)
        new_height = int(new_height)
        new_image = self.image.resize((new_width, new_height), Image.ANTIALIAS)
        new_image.save(new_image_name)

    def rescale_image_by_factor(self, scale_factor, new_image_name):
        new_width = int(self.width*scale_factor)
        new_height = int(self.height*scale_factor)
        new_image = self.image.resize((new_width, new_height), Image.ANTIALIAS)
        new_image.save(new_image_name)

    def apply_contrast(self, factor, save_name):
        enhancer = ImageEnhance.Contrast(self.image)
        new_image = enhancer.enhance(factor)
        new_image.save(save_name)

    def apply_sharpness(self, factor, save_name):
        enhancer = ImageEnhance.Sharpness(self.image)
        new_image = enhancer.enhance(factor)
        new_image.save(save_name)