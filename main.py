# noinspection PyUnresolvedReferences
# import tensorflow as tf

from Utility.image_handler import *
from UpscaleMethods.upscale_interpolation import *
from UpscaleMethods.upscale_nn_progressive import *

if __name__ == '__main__':
    im_handler = ImageHandler()
    im_handler.read_image("./data/original/owl.png")

    interpolator = Interpolator()
    interpolator.upscale_image(image_name="./data/original/owl.png",
                               distance=3, save_name="./data/test/Output_Interpolation.png")

    im_handler.rescale_image(im_handler.width*3, im_handler.height*3, "./data/test/Output_PIL.png")
    progressive_nn = ProgressiveNN(nr_channels=3)
    progressive_nn.load(agent_id=8, to_compile=False)
    # progressive_nn.train_x2()
    # progressive_nn.save()
    progressive_nn.upscale_x2("./data/original/owl.png", "./data/test/Output_NNP.png")

