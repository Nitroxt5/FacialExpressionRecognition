from PIL import Image


class ImgConverter:
    def __init__(self):
        self._size = 48
        self._shape = (self._size, self._size)

    def convert_image(self, img: Image):
        return img.resize(self._shape)
