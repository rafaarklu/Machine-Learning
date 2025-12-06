# Documentação - K-Means Clustering no Titanic

Este script aplica o algoritmo **K-Means Clustering** ao dataset Titanic, baixado automaticamente do **Kaggle** via `kagglehub`.  
O objetivo é agrupar os passageiros com base em características selecionadas e visualizar os clusters.

# Dados utilizados

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



# Introdução ao K-Means

O **K-Means** é um algoritmo de **agrupamento não supervisionado**, cujo objetivo é dividir os dados em *k* grupos (clusters) de forma que os elementos dentro de um mesmo cluster sejam semelhantes entre si e diferentes dos clusters vizinhos.

A ideia central é:

> Encontrar centros (chamados de **centroides**) que representem cada grupo, e atribuir cada ponto ao centro mais próximo.

O K-Means é amplamente utilizado em:
- segmentação de clientes  
- compressão de imagens  
- agrupamento de usuários  
- descoberta de padrões ocultos em datasets não rotulados  

Como o dataset Titanic não possui um rótulo de cluster verdadeiro, o objetivo aqui é **descobrir segmentos naturais entre os passageiros**.

---

# Como o K-Means funciona (Intuição + Matemática mínima)

## Intuição do algoritmo

O algoritmo segue duas etapas repetidas:

1. **Atribuição**: cada ponto é colocado no cluster cujo centróide é o mais próximo.  
2. **Atualização**: recalculam-se os centróides como a média dos pontos atribuídos ao cluster.

Isso é repetido até que os centróides não mudem mais ou o número máximo de iterações seja alcançado.

---
##  Limitações teóricas do K-Means

- Assume clusters **esféricos e de tamanhos semelhantes**  
- Sensível a **outliers**  
- Sensível à inicialização → por isso existe o **k-means++**  
- Requer definição prévia de **k**  
- Funciona melhor com variáveis **numéricas e escaladas**

---

# Bibliotecas utilizadas (com justificativa teórica)

- **pandas** → manipulação dos dados brutos  
- **matplotlib** → visualização e inspeção dos clusters  
- **KMeans (scikit-learn)** → implementação do algoritmo  
- **StandardScaler** → K-Means exige padronização (distâncias euclidianas)  
- **LabelEncoder** → transformação de variáveis categóricas  
- **kagglehub** → download programático do dataset  
- **StringIO** → suporte à exportação do gráfico no formato SVG  

---

# Código Exemplo e execução


=== "output"

    ``` python exec="on" html="1"
    --8<-- "./docs/k-means/kmeans_script.py"
    ```

=== "code"

    ``` python exec="off"
    --8<-- "./docs/k-means/kmeans_script.py"
    ```

# Download e carregamento do dataset

```python
path = kagglehub.dataset_download("yasserh/titanic-dataset")
file_path = os.path.join(path, "Titanic-Dataset.csv")
df = pd.read_csv(file_path)
```

---

# Preparação dos dados (com explicações teóricas)

## ✔ Por que preparar os dados?

O K-Means **não trabalha com dados ausentes, nem com variáveis categóricas**, e é sensível às escalas.  
Por isso seguimos estes passos:

---

## ✔ Seleção das variáveis

Foram escolhidas:

- Pclass  
- Sex  
- Age  
- SibSp  
- Parch  
- Fare  

Essas variáveis representam bem o perfil dos passageiros.

---

## ✔ Conversão de variável categórica

`Sex` → 0 ou 1.

---

## ✔ Tratamento de valores ausentes

`Age` → preenchimento com a **mediana**.

---

## ✔ Normalização (item crítico!)

O K-Means usa **distância euclidiana**, então variáveis com magnitudes maiores dominam o clustering.  
Por isso aplicamos `StandardScaler()`:

---

```python
features = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
features['Sex'] = LabelEncoder().fit_transform(features['Sex'])
features['Age'].fillna(features['Age'].median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
```

---

# Aplicação do algoritmo K-Means (com teoria dos parâmetros)

## Parâmetros utilizados:

- **n_clusters=2**  
- **init='k-means++'**  
- **max_iter=100**  
- **random_state=42**  
- **n_init=10**  

---

```python
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
```

---

# Visualização dos clusters

```python
plt.figure(figsize=(10, 8))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50)

plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1],
            c='red', marker='*', s=200, label='Centroids')

plt.title("K-Means Clustering - Titanic Dataset")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.legend()
```

---

# Exportação do gráfico para MkDocs

```python
from io import StringIO
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
```

---

# Interpretação dos clusters

Os clusters não representam classes "verdadeiras", mas padrões naturais dos dados.  
Com frequência, no Titanic, os agrupamentos tendem a separar:

- passageiros de classes mais altas  
- tarifas maiores  
- famílias menores  
- perfis de idade distintos  

---

