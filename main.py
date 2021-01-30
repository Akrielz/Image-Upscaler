# noinspection PyUnresolvedReferences
# import tensorflow as tf
from image_handler import *

if __name__ == '__main__':
    im_handler = ImageHandler()
    im_handler.read_image("data/original/dog_0.png")
    im_handler.downscale_image_keep_ratio_width(im_handler.width/2, "data/downscaled/dog_0_d2_1.png")
    im_handler.downscale_image_keep_ratio_height(im_handler.height/2, "data/downscaled/dog_0_d2_2.png")
    im_handler.downscale_image(im_handler.width/4, im_handler.height/3, "data/downscaled/dog_0_d2_3.png")
