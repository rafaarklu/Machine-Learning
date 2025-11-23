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




    
=== "output"

    ``` python exec="on" html="1"
    --8<-- "./docs/k-means/kmeans_script.py"
    ```

=== "code"

    ``` python exec="off"
    --8<-- "./docs/k-means/kmeans_script.py"
    ```

---

# Documentação - K-Means Clustering no Titanic

Este script aplica o algoritmo **K-Means Clustering** ao dataset Titanic, baixado automaticamente do **Kaggle** via `kagglehub`.  
O objetivo é agrupar os passageiros com base em características selecionadas e visualizar os clusters.

---

## 1. Importação das bibliotecas

- **pandas**: manipulação de dados.  
- **matplotlib**: visualização gráfica.  
- **sklearn.cluster.KMeans**: algoritmo de agrupamento.  
- **sklearn.preprocessing.StandardScaler, LabelEncoder**: pré-processamento (normalização e codificação de variáveis categóricas).  
- **kagglehub**: download automático do dataset.  
- **os**: manipulação de caminhos de arquivos.  
- **StringIO**: salvar o gráfico em formato SVG para exibição no MkDocs.

---

## 2. Download e carregamento do dataset

1. O dataset **Titanic-Dataset.csv** é baixado do Kaggle através do `kagglehub`.  
2. O arquivo é carregado em um DataFrame `pandas`.

```python exec="off"
path = kagglehub.dataset_download("yasserh/titanic-dataset")
file_path = os.path.join(path, "Titanic-Dataset.csv")
df = pd.read_csv(file_path)
```

## 3. Preparação dos dados

- Seleção de variáveis: Age e Fare.

- Codificação de variável categórica: Sex convertido para valores numéricos (0 ou 1).

- Tratamento de valores ausentes: valores nulos em Age substituídos pela mediana.

- Normalização: as variáveis são escalonadas (StandardScaler) para padronizar a magnitude.

```python exec="off"
features = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
features['Sex'] = LabelEncoder().fit_transform(features['Sex'])
features['Age'].fillna(features['Age'].median(), inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
```

## 4. Aplicação do K-Means

- Número de clusters: 2 (definido manualmente).

- Inicialização: k-means++.

- Máximo de iterações: 100.

- Semente aleatória (random_state=42) para reprodutibilidade.

- Resultado: vetor labels com o cluster de cada passageiro.

```python exec="off"
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
```

## 5. Visualização dos clusters

- Plotagem em 2D com as duas primeiras variáveis escalonadas.

- Passageiros são representados por pontos coloridos conforme o cluster.

- Centros dos clusters (centroids) destacados em vermelho (*).

```python exec="off"
plt.figure(figsize=(10, 8))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='*', s=200, label='Centroids')
plt.title("K-Means Clustering - Titanic Dataset")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.legend()
```
## 6. Exportação do gráfico para MkDocs

- O gráfico é salvo em formato SVG em memória (StringIO).

- O conteúdo é exibido via print() para integração no MkDocs.

```python exec="off"
from io import StringIO
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
```
