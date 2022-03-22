import pickle


class Classificadores:


    def __init__(self):
        self.diretorio = './classificadores/'
        print(self.diretorio)

    def carregar_classificador(self, nome_classificador):
        return pickle.load(open(self.diretorio + nome_classificador, 'rb'))

    def classificarEstagio1(self, vetor_caracteristicas):
        rotulo = 'Estágio 1'
        classificador = self.carregar_classificador('classificador_estagio_1.sav')
        resultado_classificacao = classificador.predict(vetor_caracteristicas)
        #print(resultado_classificacao)

    def classificarEstagio2(self, vetor_caracteristicas):
        self.diretorio += 'classificador_estagio_2.sav'

    def classificarEstagio3(self, vetor_caracteristicas):
        self.diretorio += 'classificador_estagio_3.sav'

    def classificarEstagio4(self, vetor_caracteristicas):
        self.diretorio += 'classificador_estagio_4.sav'

    def classificarEstagio5(self, vetor_caracteristicas):
        self.diretorio += 'classificador_estagio_5.sav'