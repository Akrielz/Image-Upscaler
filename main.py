# noinspection PyUnresolvedReferences
# import tensorflow as tf

from Utility.image_handler import *
from UpscaleMethods.upscale_interpolation import *
from UpscaleMethods.upscale_nn_progressive import *

if __name__ == '__main__':
    test_image = "./data/test/mini_d4_avatar.png"

    cse_good_agent_id = 462  # categorical_crossentropy
    mse_good_agent_id = 691  # mean_squared_error

    # im_handler = ImageHandler()
    # im_handler.read_image(test_image)
    # # im_handler.rescale_image(im_handler.width // 4, im_handler.height // 4, "./data/test/mini_d4_avatar.png")
    # im_handler.rescale_image(im_handler.width * 2, im_handler.height * 2, "./data/test/Output_PIL.png")
    # interpolator = Interpolator()
    # interpolator.upscale_image(image_name=test_image,
    #                            distance=2, save_name="./data/test/Output_Interpolation.png")

    progressive_nn = ProgressiveNN(nr_channels=3)
    progressive_nn.load(agent_id=mse_good_agent_id, to_compile=True)
    progressive_nn.train(training_directory="./data/original/wallpapers-sky", method="x8", start=0)
    # Nature:   x2 -> 35  | x4 -> 4
    # Sky:      x2 -> 159 | x4 -> 246  # ID 462

    progressive_nn.upscale_x8(test_image, "./data/test/Output_NNP_F3.png", color_approx=False)
    progressive_nn.upscale_x8(test_image, "./data/test/Output_NNP_T.png", color_approx=True)
