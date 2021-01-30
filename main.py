# noinspection PyUnresolvedReferences
# import tensorflow as tf
from image_handler import *
from upscale_interpolation import *

if __name__ == '__main__':
    im_handler = ImageHandler()
    im_handler.read_image("data/original/SorakaPrestigeHeart.png")
    interpolator = Interpolator()
    interpolator.upscale_image("data/original/SorakaPrestigeHeart.png", distance=10)
    im_handler.rescale_image(im_handler.width*10, im_handler.height*10, "temp2.png")
    # interpolator.upscale_image("temp.png", distance=2)
