import random
import numpy as np

num_entradas = 4
num_neuronios = 64

class Rede_Neural():
    def __init__(self, x, y, raquete, bias=-1, cor=(255,255,255)) -> None:
        
        self.entradas = np.array([x, y,raquete, bias])
        self.cor = cor
        self.pesos_camada_1 = [np.array([random.uniform(-1, 1) for _ in range(num_entradas)]) for _ in range(num_neuronios)]
        self.pesos_camada_2 = [np.array([random.uniform(-1, 1) for _ in range(len(self.pesos_camada_1))]) for _ in range(num_neuronios)]
        self.pesos_camada_saida = np.array([random.uniform(-1, 1) for _ in range(len(self.pesos_camada_2))])

    def feedForward(self):
        self.saida_camada_1 = []
        self.saida_camada_2 = []
        
        for i in range(len(self.pesos_camada_1)):
            self.saida_camada_1.append(round(tanh(sum(self.pesos_camada_1[i] * self.entradas)), 6))
        self.saida_camada_1 = np.array(self.saida_camada_1)
        
        for i in range(len(self.pesos_camada_2)):
            self.saida_camada_2.append(round(tanh(sum(self.pesos_camada_2[i] * self.saida_camada_1)), 6))
        self.saida_camada_2 = np.array(self.saida_camada_2)
        
        self.saida_camada_saida = round(sigmoid(sum(self.pesos_camada_saida * self.saida_camada_2)), 6)

        return self.saida_camada_saida


    def backPropagation(self, erro, taxa_aprendizagem = 0.001):
        for i in range(len(self.pesos_camada_saida)):
            self.pesos_camada_saida[i] = self.pesos_camada_saida[i] + (taxa_aprendizagem * erro * self.saida_camada_2[i])
        self.pesos_camada_saida = np.array(self.pesos_camada_saida)

        for i, n in enumerate(self.pesos_camada_2):
            for j in range(len(n)):
                self.pesos_camada_2[i][j] = self.pesos_camada_2[i][j] + (taxa_aprendizagem * erro * self.saida_camada_1[j])
        self.pesos_camada_2 = np.array(self.pesos_camada_2)

        for i, n in enumerate(self.pesos_camada_1):
            for j in range(len(n)):
                self.pesos_camada_1[i][j] = self.pesos_camada_1[i][j] + (taxa_aprendizagem * erro * self.entradas[j])
        self.pesos_camada_1 = np.array(self.pesos_camada_1)

    def atualizarEntradas(self, x, y, raquete, bias=-1):
        self.entradas = np.array([x, y,raquete, bias])

    def saida(self):
        print(self.saida_camada_saida)

    def salvarPesos(self):
        with open("dados.txt", "w") as arq:
            for i, n in enumerate(self.pesos_camada_1):
                arq.write(f"--------Camada 1--------\n")
                arq.write(f"------Neuronio {i}------\n")
                for k, j in enumerate(n):
                    arq.write(f"Peso referente a entrada {k}: {j}\n")
                arq.write(f"{10 * '---'}\n")
            arq.write(f"{20 * '---'}\n")

            for i, n in enumerate(self.pesos_camada_2):
                arq.write(f"--------Camada 2--------\n")
                arq.write(f"------Neuronio {i}------\n")
                for k, j in enumerate(n):
                    arq.write(f"Peso referente a entrada {k}: {j}\n")
                arq.write(f"{10 * '---'}\n")
            arq.write(f"{20 * '---'}\n")

            arq.write(f"------Camada Saida------\n")
            for k, j in enumerate(self.pesos_camada_saida):
                arq.write(f"Peso referente a entrada {k}: {j}\n")
                
    def getCor(self):
        return self.cor       
        

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # 0 e 1

def tanh(x):
    return np.tanh(x) # -1 , 1