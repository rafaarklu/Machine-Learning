# Machine Learning com KNN (K-Nearest Neighbors)

## Base de Dados Utilizada

A base de dados utilizada é o dataset do Titanic, obtido através da biblioteca `kagglehub`:\
https://www.kaggle.com/datasets/brendan45774/test-file

## Objetivo do Projeto

O objetivo deste trabalho é aplicar o algoritmo K-Nearest Neighbors (KNN) para resolver um problema de **Classificação Binária** e prever a **Sobrevivência (Survived)** de um passageiro no Titanic.

Para simplificar a visualização da fronteira de decisão, o modelo utiliza apenas as seguintes features contínuas:

-   **Age (Idade do Passageiro)**
-   **Fare (Preço da Passagem)**

## Análise e Modelo KNN (K=5)

O gráfico gerado abaixo representa a fronteira de decisão do modelo. As regiões coloridas indicam a classe (sobrevivência) que o modelo KNN prediz para qualquer novo passageiro que caia naquela área do plano 2D.



=== "output"

    ``` python exec="on" html="1"
    --8<-- "./docs/k-nearest-neighbor/knn_script.py"
    ```

=== "code"

    ``` python exec="off"
    --8<-- "./docs/k-nearest-neighbor/knn_script.py"
    ```
### Interpretação dos Resultados

O desempenho do modelo demonstra que a **Idade** e o **Preço da Passagem (Fare)** são preditores significativos para a sobrevivência no Titanic. Passageiros com passagens mais caras (classes mais altas) tinham maior probabilidade de sobreviver, assim como certos grupos etários.

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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler # CRÍTICO para KNN
import seaborn as sns
import pandas as pd
import kagglehub

# Carregar o dataset via kagglehub
kaggle_dataset = kagglehub.dataset_download("brendan45774/test-file")
df = pd.read_csv(f"{kaggle_dataset}/titanic.csv")

# Selecionar Features (X) e Variável Alvo (y)
X = df[['Age', 'Fare']]
y = df['Survived']  # Variável binária: 0 (Não Sobreviveu) ou 1 (Sobreviveu)

# Limpeza e Preparação
data = X.copy()
data['Survived'] = y
data = data.dropna()  # Remove valores ausentes

X_clean = data[['Age', 'Fare']]
y_clean = data['Survived']
```

### 2. Escalonamento dos Atributos (StandardScaler)

O KNN é baseado na distância. Sem esta etapa, a variável **Fare** (valores maiores) dominaria o cálculo da distância em relação à **Age** (valores menores), invalidando o modelo.

``` python
# Aplica a Padronização (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)
```

### 3. Treinamento e Avaliação do Modelo

Dividimos os dados escalonados (`X_scaled_df`) e treinamos o `KNeighborsClassifier` com **K=5 vizinhos**.

``` python
# Divisão Treino/Teste (70% Treino, 30% Teste)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y_clean,
    test_size=0.3,
    random_state=42
)

# Treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prever e Avaliar
predictions = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=['Não Sobreviveu', 'Sobreviveu']))
```

### 4. Geração da Fronteira de Decisão

Este passo gera a grade de visualização para plotar a fronteira de decisão do modelo KNN.

``` python
plt.figure(figsize=(12, 10))

# Amostragem para plotagem
sample_size = 5000
X_vis = X_scaled_df.sample(sample_size, random_state=42)
y_vis = y_clean.loc[X_vis.index]

# Definição dos limites da grade
h = 0.05
x_min, x_max = X_scaled_df.iloc[:, 0].min() - 0.5, X_scaled_df.iloc[:, 0].max() + 0.5
y_min, y_max = X_scaled_df.iloc[:, 1].min() - 0.5, X_scaled_df.iloc[:, 1].max() + 0.5

# Criação da grade e predição
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = knn.predict(grid_points).reshape(xx.shape)

# Plot da fronteira e dos dados
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu_r, alpha=0.3)
sns.scatterplot(x=X_vis.iloc[:, 0], y=X_vis.iloc[:, 1], hue=y_vis, 
                style=y_vis, palette={0: 'blue', 1: 'orange'}, 
                s=100, markers={0: 'o', 1: 'x'}, legend='full')

plt.xlabel("Age (Scaled)")
plt.ylabel("Fare (Scaled)")
plt.title("KNN Decision Boundary (k=5) - Titanic Dataset")
plt.legend(title="Titanic - Survivability", labels=["Não Sobreviveu", "Sobreviveu"])

plt.tight_layout()
plt.show()
```
