import cv2 as cv

class PreProcessamento:
    def __init__(self):
        print('INICIANDO PROCESSO DE PRÉ-PROCESSAMENTO...')

    def preProcessarImagem(self, imagem):
        imagem_rgb = self.converterBGR2RGB(imagem)
        imagem_redimensionada = self.redimensionarImagem(imagem_rgb)
        imagem_filtrada = self.filtoNLM(imagem_redimensionada, patch_size=7, patch_distance=1)
        imagem_lab = self.converterRGB2LAB(imagem_filtrada)
        return imagem_lab

    def redimensionarImagem(self, imagem, largura=3072, altura=3072):
        print(f'Rediomensionando a imagem. Largura = {largura} e altura = {altura}...')
        return cv.resize(imagem, (largura, altura))

    def filtoNLM(self, imagem_ruidosa, patch_size=7, patch_distance=11):
        print(f'Aplicando filtro Non-local Means. Patch size = {patch_size} e'
              f' patch distance = {patch_distance}...')
        return cv.fastNlMeansDenoisingColored(imagem_ruidosa, templateWindowSize=patch_size,
                                       searchWindowSize=patch_distance)

    def converterBGR2RGB(self, imagem_bgr):
        print('Convertendo imagem BGR para RGB...')
        return cv.cvtColor(imagem_bgr, cv.COLOR_BGR2RGB)

    def converterRGB2LAB(self, imagem_rgb):
        print('Convertendo imagem RGB para LAB...')
        return cv.cvtColor(imagem_rgb, cv.COLOR_RGB2LAB)

    def converterBGR2GRAY(self, imagem_bgr):
        print('Convertendo imagem BGR para escala de cinzas...')
        return cv.cvtColor(imagem_bgr, cv.COLOR_BGR2GRAY)
