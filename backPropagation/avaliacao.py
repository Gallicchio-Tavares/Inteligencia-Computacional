import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def ler_resultados(caminho_arquivo):
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        conteudo = f.read()
    
    resultados = {}
    partes = re.split(r"Resultados com (\d+) ruído?s?:", conteudo)  # Trata "ruído" e "ruídos"
    for i in range(1, len(partes), 2):  # Percorre os níveis de ruído e suas matrizes
        nivel_ruido = int(partes[i])  # Número do ruído
        matriz_texto = partes[i + 1].strip()  # Matriz correspondente
        
        # Processar matriz
        linhas_matriz = []
        for linha in matriz_texto.split("\n"):
            linha = linha.strip()
            if linha.startswith("[") and linha.endswith("]"):
                linha = linha.replace("[", "").replace("]", "").strip()  # Remove colchetes
                valores = [float(x) for x in linha.split()]  # Converte para float
                linhas_matriz.append(valores)
        matriz = np.array(linhas_matriz)  # Converte lista para NumPy array
        resultados[nivel_ruido] = matriz
    return resultados

def calcular_mse(matriz_saida, matriz_identidade): # erro quadratico medio
    return np.mean((matriz_saida - matriz_identidade) ** 2)

def calcular_acuracia(matriz_saida):
    return np.mean(np.diag(matriz_saida))

caminhos_arquivos = [
    "data/rede15/network15_lr0.1_momentum0.0.txt",
    "data/rede15/network15_lr0.4_momentum0.0.txt",
    "data/rede15/network15_lr0.9_momentum0.0.txt",
    "data/rede15/network15_lr0.1_momentum0.4.txt",
    "data/rede15/network15_lr0.9_momentum0.4.txt",
    "data/rede25/network25_lr0.1_momentum0.0.txt",
    "data/rede25/network25_lr0.4_momentum0.0.txt",
    "data/rede25/network25_lr0.9_momentum0.0.txt",
    "data/rede25/network25_lr0.1_momentum0.4.txt",
    "data/rede25/network25_lr0.9_momentum0.4.txt",    
    "data/rede35/network35_lr0.1_momentum0.0.txt",
    "data/rede35/network35_lr0.4_momentum0.0.txt",
    "data/rede35/network35_lr0.9_momentum0.0.txt",
    "data/rede35/network35_lr0.1_momentum0.4.txt",
    "data/rede35/network35_lr0.9_momentum0.4.txt",
]

matriz_identidade = np.eye(10) # a saida desejada

dados = []

# Ler os arquivos e calcular MSE e acurácia
for caminho in caminhos_arquivos:
    resultados = ler_resultados(caminho)
    nome_rede = caminho.split("/")[1]  # Obtém "rede15", "rede25" ou "rede35"
    lr = float(re.search(r"lr([\d.]+)", caminho).group(1))  # Extrai o Learning Rate
    momentum = float(re.search(r"momentum([\d.]+)(?=\.txt)", caminho).group(1))  # Extrai Momentum
    
    for nivel_ruido, matriz_saida in resultados.items():
        mse = calcular_mse(matriz_saida, matriz_identidade)
        acuracia = calcular_acuracia(matriz_saida)
        dados.append({
            "Rede": nome_rede,
            "Learning Rate": lr,
            "Momentum": momentum,
            "Nível de Ruído": nivel_ruido,
            "MSE": mse,
            "Acurácia": acuracia
        })

df = pd.DataFrame(dados)

def plotar_graficos_por_rede(df):
    estilos = {0.0: "--", 0.4: "-"} # linha diferente pra cada momentum
    cores = {0.1: "blue", 0.4: "green", 0.9: "red"}  # cores para diferentes learning rates

    for rede in df["Rede"].unique():
        plt.figure(figsize=(10, 6))
        for (lr, momentum), grupo in df[df["Rede"] == rede].groupby(["Learning Rate", "Momentum"]):
            plt.plot(
                grupo["Nível de Ruído"], 
                grupo["MSE"], 
                label=f"lr={lr}, momentum={momentum}", 
                color=cores[lr], 
                linestyle=estilos[momentum]
            )
        plt.xlabel("Nível de Ruído")
        plt.ylabel("MSE")
        plt.title(f"Comparação de MSE - {rede}")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 6))
        for (lr, momentum), grupo in df[df["Rede"] == rede].groupby(["Learning Rate", "Momentum"]):
            plt.plot(
                grupo["Nível de Ruído"], 
                grupo["Acurácia"], 
                label=f"lr={lr}, momentum={momentum}", 
                color=cores[lr], 
                linestyle=estilos[momentum]
            )
        plt.xlabel("Nível de Ruído")
        plt.ylabel("Acurácia")
        plt.title(f"Comparação de Acurácia - {rede}")
        plt.legend()
        plt.grid()
        plt.show()

plotar_graficos_por_rede(df)