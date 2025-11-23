# Projeto: Classificação de Sobreviventes do Titanic com Random Forest

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



# Código Feito


=== "output"

    ``` python exec="on" html="1"
    --8<-- "./docs/random-forest/random-forest.py"
    ```


=== "code"

    ``` python exec="off" 
    --8<-- "./docs/random-forest/random-forest.py"
    ```



## 1. Exploração dos Dados

Nesta etapa, foi realizada a análise inicial do conjunto de dados **Titanic-Dataset** (fonte: Kaggle).  
O objetivo é prever se um passageiro sobreviveu ou não, com base em suas características demográficas e socioeconômicas.

**Variáveis principais:**
- `Pclass` — Classe do bilhete (1ª, 2ª ou 3ª classe)  
- `Sex` — Sexo do passageiro  
- `Age` — Idade  
- `SibSp` — Número de irmãos/cônjuges a bordo  
- `Parch` — Número de pais/filhos a bordo  
- `Fare` — Tarifa paga  
- `Embarked` — Porto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)  
- `Survived` — Variável alvo (0 = não sobreviveu, 1 = sobreviveu)

**Estatísticas descritivas básicas:**
- Média de idade ≈ 29,6 anos  
- Tarifa média ≈ 32,2 libras  
- Proporção de sobreviventes ≈ 38,4%

O conjunto contém **891 registros**, com algumas variáveis categóricas e valores ausentes em `Age` e `Embarked`.

---

## 2. Pré-processamento

Foram realizadas as seguintes etapas de preparação dos dados:

- **Tratamento de valores ausentes:**
  - `Age`: substituído pela média das idades.  
  - `Embarked`: substituído pelo valor mais frequente (moda).

- **Codificação de variáveis categóricas:**
  - `Sex` e `Embarked` foram convertidas para formato numérico utilizando **One-Hot Encoding**.
  - A primeira categoria de cada variável foi removida para evitar multicolinearidade (`drop_first=True`).

- **Normalização:**
  - Não foi necessária, pois o modelo Random Forest não é sensível à escala dos dados.

---

## 3. Divisão dos Dados

O conjunto de dados foi dividido em:
- **Treino:** 80%  
- **Teste:** 20%

Essa divisão permite avaliar a capacidade de generalização do modelo.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 4. Treinamento do Modelo

O modelo utilizado foi o **RandomForestClassifier** do `scikit-learn`.  
Foram definidos os seguintes parâmetros:

- `n_estimators=100` → número de árvores  
- `max_depth=None` → profundidade ilimitada  
- `max_features='sqrt'` → número de variáveis consideradas em cada divisão  
- `oob_score=True` → uso de validação *out-of-bag*  
- `random_state=42` → reprodutibilidade

O modelo foi treinado com o conjunto de dados de treino (`X_train`, `y_train`).

---

## 5. Avaliação do Modelo

Após o treinamento, o modelo foi testado com o conjunto de teste (`X_test`).

**Métricas de desempenho obtidas:**
```python
Acurácia: 0.804
OOB Score: 0.794
```

**Relatório de classificação:**
```
              precision    recall  f1-score   support
           0       0.82      0.86      0.84       105
           1       0.78      0.73      0.76        74
    accuracy                           0.80       179
   macro avg       0.80      0.79      0.80       179
weighted avg       0.80      0.80      0.80       179
```

O modelo apresentou desempenho **sólido e equilibrado**, com boa generalização entre as classes.

---

## 6. Importância das Variáveis

O modelo fornece uma medida de **importância de cada variável** para as decisões da floresta:

| Variável     | Importância |
|---------------|-------------:|
| Fare          | 0.274 |
| Sex_male      | 0.267 |
| Age           | 0.255 |
| Pclass        | 0.082 |
| SibSp         | 0.050 |
| Parch         | 0.037 |
| Embarked_S    | 0.023 |
| Embarked_Q    | 0.011 |

> As variáveis **Fare**, **Sex_male** e **Age** são as mais relevantes para prever a sobrevivência.  
> Isso indica que o custo do bilhete (e, indiretamente, a classe social), o sexo e a idade foram fatores determinantes.

---

## 7. Conclusão e Possíveis Melhorias

O modelo **Random Forest** apresentou **acurácia de aproximadamente 80%**, demonstrando boa capacidade de previsão.  

**Principais conclusões:**
- Passageiros com tarifas mais altas e do sexo feminino tiveram maior chance de sobreviver.
- A idade também é um fator relevante: crianças e jovens apresentaram taxas de sobrevivência superiores.

**Possíveis melhorias futuras:**
- Aplicar **tuning de hiperparâmetros** com `GridSearchCV` ou `RandomizedSearchCV`.  
- Balancear as classes com `class_weight='balanced'` ou técnicas de oversampling (ex: SMOTE).  
- Criar variáveis derivadas (ex: `FamilySize = SibSp + Parch + 1`).  
- Avaliar outros algoritmos (XGBoost, Gradient Boosting, etc.).

---

**Autor:** Rafael Arkchimor Lucena    
**Ferramentas:** Python, Scikit-Learn, Pandas, MkDocs  
**Base de dados:** [Titanic Dataset - Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
