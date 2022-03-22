import os

import cv2 as cv
from PreProcessamento import PreProcessamento
from Segmentar import Segmnetar
from ExtrairCaracteristica import ExtrairCaracteristica
from Classificadores import Classificadores


if __name__ == '__main__':
    diretorio_imagens = '../imagens/'
    locais = ['espiga', 'folha']
    for local in locais:
        diretorio_imagem = diretorio_imagens + local + '/'
        arquivos = os.listdir(diretorio_imagens + local + '/')
        for arquivo in arquivos:
            imagem = cv.imread(diretorio_imagem + arquivo)

            pre_processamento = PreProcessamento()
            imagem_pre_processada = pre_processamento.preProcessarImagem(imagem)
            imagem_r = pre_processamento.redimensionarImagem(imagem)

            segmentacao = Segmnetar()
            imagem_segmentada = segmentacao.segmentarImagem(imagem_pre_processada, imagem_r, 'folha')
            imagem_segmentada = pre_processamento.redimensionarImagem(
                pre_processamento.converterBGR2GRAY(imagem_segmentada),
                256, 256)

            extraca_cartacteristicas = ExtrairCaracteristica()
            vetor_caracteristicas = extraca_cartacteristicas.extrairCaracteristicas(imagem_segmentada)
            extraca_cartacteristicas.salvarVetorCaracteristica(vetor_caracteristicas, arquivo,
                                                               './caracteristicas/')

            vetor_caracteristicas_pca = extraca_cartacteristicas.aplicarPCA(vetor_caracteristicas, 1)
            extraca_cartacteristicas.salvarVetorCaracteristica(vetor_caracteristicas_pca, arquivo,
                                                               './caracteristicas_pca/')

            classificador1 = Classificadores()
            classificador1.classificarEstagio1(vetor_caracteristicas_pca)

            exit('FIM')
