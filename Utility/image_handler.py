# noinspection PyUnresolvedReferences
from PIL import Image, ImageEnhance


def convert_pixel_to(pixel, nr_channels=3):
    new_pixel = [0, 0, 0, 0]

    for j, pixel_value in enumerate(pixel):
        new_pixel[j] = pixel_value

    if nr_channels == 3:
        new_pixel = (new_pixel[0], new_pixel[1], new_pixel[2])
    else:
        new_pixel = (new_pixel[0], new_pixel[1], new_pixel[2], new_pixel[3])
    return new_pixel


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

    def get_pixels_copy(self):
        width, height = self.get_size()
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

    def apply_contrast(self, factor, save_name):
        enhancer = ImageEnhance.Contrast(self.image)
        new_image = enhancer.enhance(factor)
        new_image.save(save_name)

    def apply_sharpness(self, factor, save_name):
        enhancer = ImageEnhance.Sharpness(self.image)
        new_image = enhancer.enhance(factor)
        new_image.save(save_name)