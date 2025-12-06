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



# Modelo de Árvore de Decisão (Decision Tree)

Este documento apresenta a implementação, explicação teórica e análise do modelo **Decision Tree Classifier**, aplicado ao dataset *BMW Used Cars* do Kaggle.  

---

# O que é uma Árvore de Decisão?

Uma **Árvore de Decisão** (Decision Tree) é um algoritmo de Machine Learning supervisionado que pode ser usado para:

- **Classificação** (prever categorias)
- **Regressão** (prever valores contínuos)

Ela se comporta como uma sequência de perguntas do tipo *"se... então..."*.  
Cada pergunta divide os dados em grupos mais homogêneos.  
Uma árvore é composta por:

- **Raiz (root)** → primeira divisão  
- **Nós internos** → perguntas  
- **Folhas (leaves)** → resultado final (classe prevista)

### Objetivo do algoritmo
Criar divisões que deixem os grupos **o mais puros possível**, isto é, com elementos de uma única classe.

### Métricas usadas para decidir as divisões

- **Gini Impurity**  
- **Entropia**  
Ambas medem o quanto um conjunto é "misturado" entre classes.  
Quanto **menor** esse valor, mais pura é a divisão.

Árvores têm benefícios importantes:

- Interpretabilidade (gráfico fácil de entender)
- Não exigem normalização dos dados
- Aceitam dados numéricos e categóricos (com encoding)
- Capturam relações não-lineares

---

# Execução do Script

A seguir está o código executado:

=== "output"

```python exec="on" html="1"
--8<-- "./docs/Machine-Learning/decision-script.py"
```

=== "code"

```python exec="off"
--8<-- "./docs/Machine-Learning/decision-script.py"
```

---

# Passo a Passo da Implementação

## Importar bibliotecas

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

**Explicação Teórica:**  
- `DecisionTreeClassifier` cria o modelo.  
- Árvores não exigem normalização, pois dividem os dados com base em **pontos de corte**.
- `LabelEncoder` transforma textos em números (necessário porque árvores não trabalham com strings).  
- `train_test_split` separa os dados de forma aleatória.  

---

## Carregar a base

```python
path = kagglehub.dataset_download("adityadesai13/used-car-dataset-ford-and-mercedes")

df = pd.read_csv(path + "/bmw.csv")  
x = df[['model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'tax', 'engineSize']]
```

**Explicação:**  
Carregamos o dataset BMW.  
Selecionamos atributos que influenciam o consumo do veículo.

---

## Transformar variáveis e definir o alvo

```python
label_encoder = LabelEncoder()
x['model'] = label_encoder.fit_transform(x['model'])
x['transmission'] = label_encoder.fit_transform(x['transmission'])
x['fuelType'] = label_encoder.fit_transform(x['fuelType'])  

y = df['consumo_cat'] = pd.cut(
        df['mpg'],
        bins=[0, 25, 40, 100],  
        labels=['baixo', 'medio', 'alto']
)

data = x.copy()
data['target'] = y
```

### Teoria aplicada
Árvores:

- lidam bem com variáveis categóricas **desde que convertidas para números**
- não precisam que os dados sejam escalonados
- conseguem encontrar interações automaticamente, como:
  > *alta cilindrada + câmbio automático → consumo baixo*

A variável alvo é categorizada a partir da economia de combustível (`mpg`).

---

## Remover valores ausentes

```python
data = data.dropna()

x_clean = data.drop('target', axis=1)
y_clean = data['target']
```

**Por quê?**  
Árvores não lidam com valores faltantes nativamente.  
Removemos entradas com `NaN`.

---

## Treinar e avaliar o modelo

```python
x_train, x_test, y_train, y_test = train_test_split(
    x_clean, y_clean, 
    test_size=0.7, 
    random_state=42
)

classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

accuracy = classifier.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

tree.plot_tree(classifier)

buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```

### Teoria da Árvore aplicada ao treino

Árvore de decisão:

1. **Seleciona uma variável**
2. **Testa possíveis cortes**
3. **Calcula o ganho de impureza**
4. **Escolhe o melhor ponto**
5. Repete o processo até:
   - atingir profundidade máxima
   - ou não haver melhora

Como não definimos parâmetros, o modelo:

- cresce até o limite (tendência a overfitting)
- usa Gini Impurity por padrão

---

# Interpretação dos Resultados

A acurácia indica o quão bem o modelo classificou as categorias de consumo (`baixo`, `médio`, `alto`).

### Ponto forte das árvores
Elas conseguem aprender regras como:

- **Preço alto + motor grande → consumo baixo**
- **Carro novo + motor pequeno → consumo alto**
- **Alta quilometragem → pior consumo**  

Essas regras são automaticamente encontradas pelo algoritmo.

### Fragilidade
Sem limitar:
- profundidade  
- número mínimo de amostras por nó  
a árvore pode decorar o treino → **overfitting**.

---

# Conclusão

Este projeto mostra:

- Como aplicar uma **Árvore de Decisão** na prática  
- Como transformar variáveis categóricas  
- Como criar uma variável alvo categorizada  
- Como visualizar e interpretar o modelo  
- Como avaliar desempenho  

Árvores são um dos algoritmos mais intuitivos e importantes do Machine Learning moderno, base de modelos mais poderosos como Random Forest e Gradient Boosting.

