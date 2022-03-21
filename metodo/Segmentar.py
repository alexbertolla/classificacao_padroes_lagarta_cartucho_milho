import numpy as np
import cv2


class Segmnetar:
    def __init__(self):
        print('INICIANDO PROCESSO DE SEGMENTAÇÃO...')

    def segmentacaoMorfologica(self, imagem_pre_segmentada):
        janela = np.ones((5, 5), np.uint8)
        imagem_segmentada_morfologica = cv2.morphologyEx(imagem_pre_segmentada, cv2.MORPH_CLOSE, kernel=janela,
                                                         iterations=3)
        return imagem_segmentada_morfologica

    def segmentarPelaImagemBinaria(self, imagem, imagem_binaria):
        for l in range(len(imagem_binaria)):
            for c in range(len(imagem_binaria)):
                if not imagem_binaria[l, c]:
                    imagem[l, c, :] = 0
        return imagem

    def segmentarImagem(self, imagem_lab, imagem_original, localizacao):
        l, a, b = cv2.split(imagem_lab)
        a = self.segmentacaoMorfologica(a)
        b = self.segmentacaoMorfologica(b)

        limiar_a, binaria_a = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        limiar_b, binaria_b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if localizacao == 'folha':
            return self.segmentarEmFolha(imagem_original, a, b, limiar_a, limiar_b)
        elif localizacao == 'espiga':
            return self.segmentarEmEspiga(imagem_original, b, limiar_b)
        else:
            exit(f'ERRO: LOCALIZAÇÃO INVÁLIDA \'{localizacao}\'')

    def segmentarEmFolha(self, imagem_original, componente_a, componente_b, limiar_a, limiar_b):
        print('SEGMENTANDO IMAGEM DA LAGARTA PRESENTE EM FOLHA...')
        imagem_binaria_a = componente_a > limiar_a
        imagem_binaria_b = componente_b < limiar_b
        imagem_binaria_a_e_b = np.bitwise_and(imagem_binaria_a, imagem_binaria_b)

        imagem_segmentada = self.segmentarPelaImagemBinaria(imagem_original, imagem_binaria_a_e_b)
        return imagem_segmentada

    def segmentarEmEspiga(self, imagem_original, componente_b, limiar_b):
        print('SEGMENTANDO IMAGEM DA LAGARTA PRESENTE EM ESPIGA...')
        sigma = 0.25
        desvio_padrao_b = int(np.std(componente_b))

        b_abaixo_limiar = componente_b < limiar_b - (sigma * desvio_padrao_b)
        imagem_segmentada = self.segmentarPelaImagemBinaria(imagem_original, b_abaixo_limiar)
        return imagem_segmentada
