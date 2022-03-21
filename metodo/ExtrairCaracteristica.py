import numpy as np
import cv2
from sklearn.decomposition import PCA
from skimage.feature import hog
from math import copysign, log10


class ExtrairCaracteristica:

    def __init__(self):
        print('INICIANDO PROCESSO DE EXTRAÇÃO DE CARACTERÍSTICAS...')

    def normalizarVetor(self, vetor_caracteristicas):
        print('NORMALIZANDO VETOR DE CARACTERÍSTICAS...')
        media = np.mean(vetor_caracteristicas)
        desvio_padrao = np.std(vetor_caracteristicas)
        vetor_normalizado = np.copy(vetor_caracteristicas)

        linha = 1
        for caracteristica in range(len(vetor_caracteristicas)):
            valor = vetor_caracteristicas[caracteristica]
            z = (valor - media) / desvio_padrao
            vetor_normalizado[caracteristica] = z
            linha += 1
        return vetor_normalizado

    def aplicarPCA(self, vetor_caracteristicas, n_components=128):
        print('APLICANDO PCA NO VETOR DE CARACTERÍSTICAS...')
        vetor_caracteristicas = np.vstack(vetor_caracteristicas).astype(np.float64)
        pca = PCA(n_components=n_components)
        pca.fit(vetor_caracteristicas)
        vetor_pca = pca.transform(vetor_caracteristicas)
        return vetor_pca

    def salvarVetorCaracteristica(self, vetor_caracteristicas, nome_imagem, diretorio):
        print('SALVANDO VETOR DE CARACTERÍSTICAS...')

        python_file = open(diretorio + nome_imagem + '.txt', "w")
        linha = ''
        for caracteristica in vetor_caracteristicas:
            linha += str(caracteristica) + '\n'
        python_file.write(linha)
        python_file.close()


    def extrairHOG(self, imagem):
        print('EXTRAINDO CARACTERÍSTICAS HOG...')
        caracteristicas, imagem_hog = hog(imagem,
                                          orientations=9,
                                          pixels_per_cell=(16, 16),
                                          cells_per_block=(2, 2),
                                          transform_sqrt=False,
                                          visualize=True,
                                          feature_vector=True)
        return caracteristicas

    def extrairHu(self, imagem):
        print('EXTRAINDO MOMENTOS INVARIANTES DE HU...')
        momentos_centrais = cv2.moments(imagem)
        momentos_hu = cv2.HuMoments(momentos_centrais)
        vetor_hu = []
        for i in range(len(momentos_hu)):
            momentos_hu[i] = -1 * copysign(1.0, momentos_hu[i]) * log10(abs(momentos_hu[i]))
            vetor_hu.append(momentos_hu[i])
        return momentos_hu

    def extrairCaracteristicas(self, imagem_segmentada):
        caracteristicas_hog = self.extrairHOG(imagem_segmentada)
        caracteristicas_hu = self.extrairHu(imagem_segmentada)

        vetor_hog_hu = np.append(caracteristicas_hog, caracteristicas_hu)
        return self.normalizarVetor(vetor_hog_hu)

