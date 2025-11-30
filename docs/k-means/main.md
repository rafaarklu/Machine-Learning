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



# Gráfico
    
=== "output"

    ``` python exec="on" html="1"
    --8<-- "./docs/k-means/kmeans_script.py"
    ```

=== "code"

    ``` python exec="off"
    --8<-- "./docs/k-means/kmeans_script.py"
    ```

---

## 1. Importação das bibliotecas

-   **pandas**: manipulação e leitura dos dados.\
-   **matplotlib**: criação de gráficos.\
-   **sklearn.cluster.KMeans**: implementação do algoritmo de
    agrupamento K-Means.\
-   **sklearn.preprocessing.StandardScaler, LabelEncoder**: padronização
    e codificação de variáveis.\
-   **kagglehub**: download automático do dataset a partir do Kaggle.\
-   **os**: manipulação de caminhos de diretórios.\
-   **StringIO**: permite exportar gráficos em SVG para uso no MkDocs.\
-   **numpy**: operações matemáticas.\
-   **sklearn.metrics**: cálculo de acurácia e matriz de confusão para
    avaliar o agrupamento.

------------------------------------------------------------------------

## 2. Download e carregamento do dataset

1.  O dataset **Titanic-Dataset.csv** é baixado automaticamente a partir
    do Kaggle utilizando `kagglehub`.\
2.  Em seguida, o arquivo é carregado em um DataFrame Pandas.

``` python
path = kagglehub.dataset_download("yasserh/titanic-dataset")
file_path = os.path.join(path, "Titanic-Dataset.csv")
df = pd.read_csv(file_path)
```

------------------------------------------------------------------------

## 3. Preparação dos dados

### Variáveis selecionadas

As seguintes variáveis são utilizadas como entrada do K-Means:

-   Pclass\
-   Sex\
-   Age\
-   SibSp\
-   Parch\
-   Fare

### Pré-processamento aplicado

-   **Codificação**: a variável `Sex` é convertida para valores
    numéricos (0/1).\
-   **Tratamento de nulos**: valores ausentes na idade (`Age`) são
    substituídos pela mediana.\
-   **Padronização (normalização)**: todas as variáveis são escalonadas
    com `StandardScaler`.\
-   **Label verdadeiro**: a variável `Survived` é guardada para comparar
    depois.

``` python
features = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
labels_true = df['Survived'].copy()

features['Sex'] = LabelEncoder().fit_transform(features['Sex'])
features['Age'].fillna(features['Age'].median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
```

------------------------------------------------------------------------

## 4. Aplicação do K-Means

-   **Clusters**: 2\
-   **Inicialização**: k-means++\
-   **Iterações máximas**: 100\
-   **n_init**: 10\
-   **Seed**: 42

``` python
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100,
                random_state=42, n_init=10)
labels_pred = kmeans.fit_predict(X_scaled)
```

------------------------------------------------------------------------

## 5. Ajuste dos rótulos

O K-Means não sabe o que representa "0" ou "1".\
Por isso, verifica-se se os rótulos precisam ser invertidos:

``` python
if accuracy_score(labels_true, labels_pred) < accuracy_score(labels_true, 1-labels_pred):
    labels_pred = 1-labels_pred
```

