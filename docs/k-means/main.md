# Documentação - K-Means Clustering no Titanic

Este script aplica o algoritmo **K-Means Clustering** ao dataset Titanic, baixado automaticamente do **Kaggle** via `kagglehub`.  
O objetivo é agrupar os passageiros com base em características selecionadas e visualizar os clusters.




    
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
