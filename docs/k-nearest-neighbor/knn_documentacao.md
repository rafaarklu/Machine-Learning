# Machine Learning com KNN (K-Nearest Neighbors)

## Base de Dados Utilizada

A base de dados utilizada é o conjunto de carros usados da BMW:\
https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes/data

## Objetivo do Projeto

O objetivo deste trabalho é aplicar o algoritmo K-Nearest Neighbors
(KNN) para resolver um problema de **Classificação** e prever o **Tipo
de Combustível (fuelType)** de um veículo.

Para simplificar a visualização da fronteira de decisão, o modelo
utiliza apenas as seguintes features contínuas:

-   **Engine Size (Tamanho do Motor)**
-   **Mileage (Quilometragem/Rodagem)**

## Análise e Modelo KNN (K=3)

O gráfico gerado abaixo representa a fronteira de decisão do modelo. As
regiões coloridas indicam a classe (tipo de combustível) que o modelo
KNN prevê para qualquer novo carro que caia naquela área do plano 2D.

  Tipo de Combustível   Cor de Classificação
  --------------------- ----------------------
  Diesel                Azul
  Petrol                Vermelho/Laranja
  Hybrid                Roxo (áreas menores)

### Exportar para as Planilhas

O alto desempenho do modelo (**acurácia esperada acima de 0.90**)
demonstra que o `Engine Size` e a `Mileage` são preditores fortes para o
`fuelType` dentro do dataset da BMW.

=== "output"

`python exec="on" html="1" --8<-- "./docs/k-nearest-neighbor/knn_script.py"`

=== "code"

`python exec="off" --8<-- "./docs/k-nearest-neighbor/knn_script.py"`

## Passo a Passo da Implementação

### 1. Importação de Bibliotecas e Carregamento de Dados

Importamos o **StandardScaler**, que é essencial para o algoritmo KNN.

``` python
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler # CRÍTICO para KNN
import seaborn as sns
import pandas as pd
import kagglehub

# Carregar o dataset
file_name = "bmw.csv"
df = pd.read_csv(file_name) 

# Selecionar Features (X) e Variável Alvo (y)
X = df[['engineSize', 'mileage']]
y, fuel_labels = pd.factorize(df['fuelType']) # fuel_labels guarda os nomes das classes

# Limpeza e Preparação
data = X.copy()
data['target'] = y
data = data.dropna() 

X_clean = data[['engineSize', 'mileage']]
y_clean = data['target']
```

### 2. Escalonamento dos Atributos (StandardScaler)

O KNN é baseado na distância. Sem esta etapa, a variável **mileage**
(valores na casa dos milhares) dominaria o cálculo da distância em
relação ao **engineSize** (valores na casa das unidades), invalidando o
modelo.

``` python
# Aplica a Padronização (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)
```

### 3. Treinamento e Avaliação do Modelo

Dividimos os dados escalonados (`X_scaled_df`) e treinamos o
`KNeighborsClassifier` com **K=3 vizinhos**.

``` python
# Divisão Treino/Teste (70% Treino, 30% Teste)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y_clean,
    test_size=0.3,
    random_state=42
)
 
# Treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Prever e Avaliar
predictions = knn.predict(X_test)
print(f"Accuracy (com escalonamento): {accuracy_score(y_test, predictions):.2f}")
```

### 4. Geração da Fronteira de Decisão

Este passo gera a grade de visualização, garantindo que o labels da
legenda seja corretamente convertido para lista para evitar o erro:
`ValueError: The truth value of a Index is ambiguous`.

``` python
plt.figure(figsize=(12, 10))

# Amostragem para plotagem e definição dos limites no espaço escalonado
sample_size = 5000
X_vis = X_scaled_df.sample(sample_size, random_state=42)
y_vis = y_clean.loc[X_vis.index]

# Definição dos limites da grade (h=0.05 é o passo no espaço normalizado)
h = 0.05 
x_min, x_max = X_scaled_df['engineSize'].min() - 0.5, X_scaled_df['engineSize'].max() + 0.5
y_min, y_max = X_scaled_df['mileage'].min() - 0.5, X_scaled_df['mileage'].max() + 0.5

# Criação da grade e predição
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = knn.predict(grid_points).reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(x=X_vis['engineSize'], y=X_vis['mileage'], hue=y_vis, style=y_vis,
                palette="deep", s=100, legend='full')

# Correção do Erro: Conversão para lista
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=fuel_labels.tolist(), title="Fuel Type")

plt.xlabel("Engine Size (Scaled)")
plt.ylabel("Mileage (Scaled)")
plt.title("KNN Decision Boundary (k=3) on Scaled Data")

# Salva o resultado
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
# print(buffer.getvalue())
```
