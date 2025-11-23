# Machine Learning com KNN (K-Nearest Neighbors)

## Base de Dados Utilizada

[Dataset - Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)


=== "PassengerId"

    Tipo: numérica discreta
    O que é: identificador único do passageiro.
    Para que serve: não contém informação útil para predição.
    Ação necessária: remover do modelo.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_titanic/passid.py"
    ```

=== "Survived"

    Tipo: categórica binária (target)
    O que é: 1 = sobreviveu; 0 = não sobreviveu.
    Para que serve: variável dependente a ser prevista.
    Ação necessária: checar balanceamento (a classe 0 é ligeiramente maior).

    ```python exec="on" html="1"
    --8<-- "docs/grafico_titanic/survived.py"
    ```

=== "Pclass"

    Tipo: categórica ordinal
    O que é: classe socioeconômica do ticket (1ª, 2ª, 3ª).
    Para que serve: proxy de condição financeira/social que influencia sobrevivência.
    Ação necessária: manter como categórica ordinal; verificar distribuição nas classes.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_titanic/pclass.py"
    ```




=== "Sex"

    Tipo: categórica binária
    Oque é: sexo biológico do passageiro (male/female).
    Para que serve: uma das variáveis mais importantes na sobrevivência.
    Ação necessária: codificar para dummy (female → 1, male → 0).

    ```python exec="on" html="1"
    --8<-- "docs/grafico_titanic/sex.py"
    ```

=== "Age"

    Tipo: numérica contínua
    O que é: idade em anos.
    Para que serve: importante para separar grupos vulneráveis (crianças, adultos).
    Ação necessária: 177 valores ausentes → imputar (média/mediana ou por título extraído do Name).

    ```python exec="on" html="1"
    --8<-- "docs/grafico_titanic/age.py"
    ```

=== "SibSp"

    Tipo: numérica discreta
    O que é: número de irmãos/cônjuges a bordo.
    Para que serve: indica tamanho do grupo familiar; pode influenciar sobrevivência.
    Ação necessária: manter; possível normalizar ou agrupar faixas.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_titanic/age.py"
    ```


=== "Parch"

    Tipo: numérica discreta
    O que é: número de pais/filhos a bordo.
    Para que serve: outro indicador do grupo familiar.
    Ação necessária: manter; possível criar “FamilySize = SibSp + Parch + 1”.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_titanic/parch.py"
    ```

=== "Ticket"

    Tipo: categórica (texto)
    O que é: número/código do ticket.
    Para que serve: pouco útil originalmente; pode ajudar se agrupado por prefixos.
    Ação necessária: normalmente remover.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_titanic/ticket.py"
    ```


=== "Fare"

    Tipo: numérica contínua
    O que é: tarifa paga pelo ticket.
    Para que serve: relação com classe social; boa variável preditiva.
    Ação necessária: checar outliers; possível normalização logarítmica.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_titanic/fare.py"
    ```


=== "Embarked"

    Tipo: categórica nominal
    O que é: porto de embarque (C, Q, S).
    Para que serve: pode refletir diferenças sociais/regionais.
    Ação necessária: imputar os 2 valores ausentes; criar dummies.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_titanic/embarked.py"
    ```


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
