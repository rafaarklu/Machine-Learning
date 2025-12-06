import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import numpy as np

# O arquivo 'bmw.csv' está no diretório atual
df = pd.read_csv("bmw.csv")

# Remove nulos da coluna 'price' (preço), que é a variável chave que vamos analisar
preco = df["price"].dropna()

# Calculando o número de bins (barras) para o histograma.
# A regra da raiz quadrada (sqrt) é um bom ponto de partida para dados reais.
bins = int(np.sqrt(len(preco)))

# Criação da figura e eixos
fig, ax = plt.subplots(figsize=(10, 6))

# Plotando o Histograma
ax.hist(
    preco, 
    bins=bins, 
    color="#1f77b4",  # Azul escuro
    edgecolor="black", 
    alpha=0.9
)

# Configurando o título e rótulos
ax.set_title("Distribuição do Preço dos Carros BMW", fontsize=16, pad=15)
ax.set_xlabel("Preço (£)", fontsize=12) 
ax.set_ylabel("Frequência", fontsize=12)

# Adicionando e configurando a grade
plt.grid(axis='y', alpha=0.5, linestyle='--')
plt.ticklabel_format(style='plain', axis='x') # Evita notação científica no eixo X

# Configuração para salvar o SVG na memória e imprimir o conteúdo (como solicitado no script anexo)
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())