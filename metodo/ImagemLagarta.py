from skimage import img_as_ubyte, img_as_float
from skimage.io import imread
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.transform import resize

class Imagem:

    def __init__(self, caminho_imagem):
        self.imagem = self.ubyte2Float(imread(caminho_imagem))


    def ubyte2Float(self, imagem_u8):
        return img_as_float(imagem_u8)

    def float2Ubyte(self, imagem_f):
        return img_as_ubyte(imagem_f)

    def rgb2Gray(self, imagem_rgb):
        return rgb2gray(imagem_rgb)

    def rgb2Hsv(self, imagem_rgb):
        return rgb2hsv(imagem_rgb)

    def rgb2Lab(self, imagem_rgb):
        return rgb2lab(imagem_rgb)


