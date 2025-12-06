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

# Random Forest

O objetivo deste documento é **explicar o funcionamento teórico do algoritmo Random Forest**, relacionando-o com a **implementação prática** aplicada ao Titanic Dataset (Kaggle).

---

# O que é Random Forest? (Explicação Teórica)

O **Random Forest** é um algoritmo de aprendizado supervisionado do tipo *ensemble*, baseado em **várias árvores de decisão**.

Ele funciona criando **múltiplas árvores de decisão independentes** e combinando seus resultados por meio de:

- **Voting** (para classificação)
- **Averaging** (para regressão)

##  Vantagens do Random Forest
- Reduz o risco de overfitting comparado a uma única árvore.
- Funciona bem com dados numéricos e categóricos.
- Tolera dados ruidosos.
- Não exige normalização ou padronização.
- Mede automaticamente a **importância das variáveis**.

## Principais conceitos que fazem o Random Forest funcionar

### *Bootstrap Aggregation* (Bagging)
Cada árvore é treinada em um **subconjunto amostrado com reposição** do dataset original.

Isso cria diversidade entre as árvores.

### Seleção aleatória de atributos
Ao construir cada divisão na árvore, o algoritmo considera **apenas um subconjunto aleatório de variáveis**.

Isso evita que todas as árvores sejam iguais ➜ aumenta a generalização.

### Out-of-Bag Score (OOB)
Como parte dos dados não entra na amostra do bootstrap, eles são usados como **validação interna**, dispensando validação cruzada.

---

# Exploração dos Dados (Aplicação Prática)

## Variáveis principais analisadas:
- `Pclass` 
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `Embarked`
- `Survived` (variável alvo)

---

# Pré-processamento (Prática + Explicação Teórica)

O Random Forest não exige padronização, porém exige:

    - Tratamento de valores ausentes
    - Codificação de variáveis categóricas (Aplicado One-Hot Encoding)



---

# Código Exemplo e resultado

=== "output"

    ``` python exec="on" html="1"
    --8<-- "./docs/random-forest/random-forest.py"
    ```


=== "code"

    ``` python exec="off" 
    --8<-- "./docs/random-forest/random-forest.py"
    ```


---
# Divisão dos Dados

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- **80% treino**
- **20% teste**

---

# Treinamento do Modelo (Prática + Explicação Teórica)

Foi utilizado o **RandomForestClassifier**, com os seguintes hiperparâmetros:

### Parâmetros utilizados
- `n_estimators=100` → 100 árvores
- `max_depth=None` → árvores podem crescer livremente
- `max_features='sqrt'` → raiz quadrada do número de variáveis
- `oob_score=True` → ativação da validação Out-of-Bag
- `random_state=42` → garante reprodutibilidade

### Por que `max_features='sqrt'`?
Essa é uma estratégia padrão para:
- aumentar a diversidade entre as árvores
- reduzir correlação entre elas
- melhorar a generalização

---

# Avaliação do Modelo

### Métricas obtidas
```
Acurácia: 0.804
OOB Score: 0.794
```

O OOB Score próximo da acurácia indica que:
- o modelo não está sofrendo overfitting
- a validade interna é consistente

### Relatório de classificação
```
              precision    recall  f1-score   support
           0       0.82      0.86      0.84       105
           1       0.78      0.73      0.76        74
    accuracy                           0.80       179
   macro avg       0.80      0.79      0.80       179
weighted avg       0.80      0.80      0.80       179
```

As classes estão relativamente equilibradas nos resultados.

---

# Importância das Variáveis (Teoria + Prática)

O Random Forest mede importância das variáveis com base em:
- redução média da impureza (Gini ou Entropia)
- ou permutação de variáveis (quando configurado)

### Importâncias obtidas
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

### Interpretação
- **Fare** (valor da passagem) → principal indicador  
  Passageiros de classes mais altas sobreviveram mais.
- **Sex_male** → homens tiveram menor chance de sobreviver.
- **Age** → crianças tiveram prioridade em barcos.
- Outros atributos tiveram menor influência.

---

# Conclusão

O modelo apresentou:
- **Acurácia ≈ 80%**
- Excelente estabilidade (OOB Score similar)
- Boa capacidade de generalização

### Conclusões importantes sobre o Titanic
- Mulheres e crianças tiveram maior probabilidade de sobreviver.
- Passagens mais caras → maior taxa de sobrevivência.
- Variáveis como `SibSp`, `Embarked` e `Parch` tiveram menor influência no modelo.

---

