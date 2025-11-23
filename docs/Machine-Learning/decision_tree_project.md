---
title: "Machine Learning com Árvore de Decisão"
author: "Documentação do Projeto"
date: "`r Sys.Date()`"
output: html_document
---

# Machine Learning com Árvore de Decisão

## Dados Utilizados
[Dataset - Kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes/data)

Utilizaremos a base de dados de **carros usados da BMW** para prever a **categoria de consumo de combustível** (*baixo, médio ou alto*) com base nas características do veículo.


=== "model"

    Tipo: categórica nominal
    O que é: modelo do carro (1 Series, 3 Series, 5 Series etc.).
    Para que serve: variável altamente relevante para preço.
    Ação necessária: transformar em dummies; verificar categorias muito raras.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_bmw/model.py"
    ```


=== "year"

    Tipo: numérica discreta
    O que é: ano de fabricação.
    Para que serve: influencia fortemente o preço devido à depreciação.
    Ação necessária: manter; possível criar “idade = ano_atual – year”.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_bmw/year.py"
    ```


=== "price"

    Tipo: numérica contínua (target típica)
    O que é: preço do carro em dólares/libra (dependendo da base).
    Para que serve: variável dependente caso o modelo seja regressão.
    Ação necessária: checar outliers e distribuição; possível usar log-transform.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_bmw/price.py"
    ```


=== "transmission"

    Tipo: categórica nominal
    O que é: tipo de transmissão (Manual, Automatic, Semi-Auto).
    Para que serve: algumas transmissões valorizam/desvalorizam o veículo.
    Ação necessária: transformar em dummies; checar categorias pouco frequentes.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_bmw/transmission.py"
    ```

    


=== "mileage"

    Tipo: numérica contínua
    O que é: quilometragem rodada.
    Para que serve: altamente correlacionado ao preço (quanto maior, menor o valor).
    Ação necessária: checar outliers; normalização opcional.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_bmw/mileage.py"
    ```


=== "fuelType"

    Tipo: categórica nominal
    O que é: tipo de combustível (Diesel, Petrol, Hybrid etc.).
    Para que serve: tem impacto direto no consumo, custo de manutenção e preço.
    Ação necessária: dummies; verificar equilíbrio entre categorias.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_bmw/fueltype.py"
    ```

=== "tax"

    Tipo: numérica contínua
    O que é: imposto anual do veículo.
    Para que serve: relacionado ao consumo/emissão; pode influenciar o valor.
    Ação necessária: manter; tratar outliers.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_bmw/tax.py"
    ```


=== "mpg"

    Tipo: numérica contínua
    O que é: consumo em milhas por galão.
    Para que serve: eficiência energética; compradores valorizam modelos econômicos.
    Ação necessária: manter; verificar valores absurdos.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_bmw/mpg.py"
    ```

=== "engineSize"

    Tipo: numérica contínua
    O que é: tamanho do motor em litros.
    Para que serve: impacta desempenho e consumo; influencia preço.
    Ação necessária: manter; checar valores inconsistentes.

    ```python exec="on" html="1"
    --8<-- "docs/grafico_bmw/enginesize.py"
    ```



---

## Modelo Árvore de Decisão (Decision Tree)


=== "output"

    ``` python exec="on" html="1"
    --8<-- "./docs/Machine-Learning/decision-script.py"
    ```

=== "code"

    ``` python exec="off" 
    --8<-- "./docs/Machine-Learning/decision-script.py"
    ```

---


## Passo a Passo

### 1. Baixar e importar bibliotecas

```python
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import kagglehub

plt.figure(figsize=(12, 10))
```

**Explicação:**  
Aqui importamos as bibliotecas necessárias:
- `pandas` para manipulação de dados,  
- `matplotlib` para visualização,  
- `scikit-learn` (`tree`, `train_test_split`, `LabelEncoder`, `accuracy_score`) para o modelo de árvore e métricas,  
- `kagglehub` para baixar o dataset,  
- `StringIO` para salvar e exibir gráficos.  

---

### 2. Carregar a base de dados

```python
path = kagglehub.dataset_download("adityadesai13/used-car-dataset-ford-and-mercedes")

df = pd.read_csv(path + "/bmw.csv")  
x = df[['model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'tax', 'engineSize']]
```

**Explicação:**  
O dataset da BMW é baixado diretamente do Kaggle e carregado em um DataFrame `df`.  
Selecionamos apenas as colunas relevantes que serão usadas como variáveis explicativas (features).

---

### 3. Transformar variáveis e criar a variável alvo

```python
label_encoder = LabelEncoder()
x['model'] = label_encoder.fit_transform(x['model'])
x['transmission'] = label_encoder.fit_transform(x['transmission'])
x['fuelType'] = label_encoder.fit_transform(x['fuelType'])  

# Definir saída (Alvo de Classificação: Categoria de Consumo)
y = df['consumo_cat'] = pd.cut(
        df['mpg'],
        bins=[0, 25, 40, 100],  
        labels=['baixo', 'medio', 'alto']
)

data = x.copy()
data['target'] = y
```

**Explicação:**  
As variáveis **categóricas** (`model`, `transmission`, `fuelType`) são convertidas para valores **numéricos** usando `LabelEncoder`.  
A variável alvo (`consumo_cat`) é criada a partir da coluna `mpg`, categorizando o consumo em **baixo, médio e alto**.  
O DataFrame final (`data`) contém tanto os atributos quanto o target.

---

### 4. Limpar valores ausentes

```python
data = data.dropna()

x_clean = data.drop('target', axis=1)
y_clean = data['target']
```

**Explicação:**  
Aqui removemos valores ausentes (`NaN`) para garantir a qualidade dos dados.  
`x_clean` contém apenas as features e `y_clean` a variável alvo (`consumo_cat`).

---

### 5. Treinar e Avaliar o Modelo

```python
x_train, x_test, y_train, y_test = train_test_split(
    x_clean, y_clean, 
    test_size=0.7, 
    random_state=42
)

# Criar e treinar o modelo
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Avaliar o modelo
accuracy = classifier.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Plotar a árvore
tree.plot_tree(classifier)

# Salvar e imprimir para exibição (útil para documentação)
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```

**Explicação:**  
1. **Divisão da base**: `train_test_split` divide os dados em treino (30%) e teste (70%).  
2. **Criação do modelo**: `DecisionTreeClassifier` é instanciado.  
3. **Treinamento**: `fit(x_train, y_train)` ajusta o modelo aos dados de treino.  
4. **Avaliação**: `score(x_test, y_test)` retorna a **acurácia** do modelo.  
5. **Visualização**: `tree.plot_tree` gera o gráfico da árvore de decisão.  
6. **Exportação**: o gráfico é salvo em formato SVG para uso em documentação.

---

# Resumo
O projeto segue as seguintes etapas:  
1. Importação de bibliotecas,  
2. Carregamento da base de dados,  
3. Transformação e criação da variável alvo,  
4. Limpeza de valores ausentes,  
5. Treinamento, avaliação e visualização da árvore de decisão.  

Esse fluxo permite **analisar o desempenho do modelo** e entender como as variáveis influenciam a categoria de consumo de combustível.

    ``` python exec="off" 
    --8<-- "./docs/Machine-Learning/decision-script.py"
    ```

---

## Passo a Passo

### 1. Baixar e importar bibliotecas

```python
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import kagglehub

plt.figure(figsize=(12, 10))
```

### 2. Carregar a base de dados

```python
path = kagglehub.dataset_download("adityadesai13/used-car-dataset-ford-and-mercedes")

df = pd.read_csv(path + "/bmw.csv")  
x = df[['model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'tax', 'engineSize']]
```

### 3. Transformar variáveis e criar a variável alvo

```python
label_encoder = LabelEncoder()
x['model'] = label_encoder.fit_transform(x['model'])
x['transmission'] = label_encoder.fit_transform(x['transmission'])
x['fuelType'] = label_encoder.fit_transform(x['fuelType'])  

# Definir saída (Alvo de Classificação: Categoria de Consumo)
y = df['consumo_cat'] = pd.cut(
        df['mpg'],
        bins=[0, 25, 40, 100],  
        labels=['baixo', 'medio', 'alto']
)

data = x.copy()
data['target'] = y
```

### 4. Limpar valores ausentes

```python
data = data.dropna()

x_clean = data.drop('target', axis=1)
y_clean = data['target']
```

### 5. Treinar e Avaliar o Modelo

```python
x_train, x_test, y_train, y_test = train_test_split(
    x_clean, y_clean, 
    test_size=0.7, 
    random_state=42
)

# Criar e treinar o modelo
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Avaliar o modelo
accuracy = classifier.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Plotar a árvore
tree.plot_tree(classifier)

# Salvar e imprimir para exibição (útil para documentação)
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```
