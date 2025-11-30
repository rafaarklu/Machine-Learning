# K-Means

=== "output"

    ``` python exec="on" html="1"
    --8<-- "./docs/k-means/kmeans_script.py"
    ```

=== "code"

    ``` python exec="off"
    --8<-- "./docs/k-means/kmeans_script.py"
    ```
---

# KNN


=== "output"

    ``` python exec="on" html="1"
    --8<-- "./docs/k-nearest-neighbor/knn_script.py"
    ```

=== "code"

    ``` python exec="off"
    --8<-- "./docs/k-nearest-neighbor/knn_script.py"
    ```

# Avaliação


## Introdução

O objetivo desta atividade é utilizar dois algoritmos de Machine
Learning --- **KNN (K-Nearest Neighbors)** e **K-Means Clustering** ---
para prever uma variável categórica.\
O dataset escolhido foi o **Titanic**, cujo objetivo principal é prever
a variável **Survived**, que indica se um passageiro sobreviveu (1) ou
não (0).

------------------------------------------------------------------------

## 1. Dataset Utilizado

O dataset do Titanic contém informações como idade, tarifa paga, classe,
sexo, entre outros atributos.\
Para este projeto, utilizamos especialmente as variáveis **Age** e
**Fare**, devidamente normalizadas, para visualizar o comportamento dos
algoritmos.

------------------------------------------------------------------------

## 2. Modelo K-Means

### 2.1 Objetivo

Embora o K-Means seja um algoritmo **não supervisionado**, ele pode ser
usado para tentar identificar agrupamentos que se aproximem da variável
categórica "Survived".

### 2.2 Resultados

-   **Acurácia**: **0.67**\

-   **Matriz de Confusão**:

        [[450  99]
         [191 151]]

### 2.3 Interpretação

-   O K-Means conseguiu formar dois clusters que, parcialmente, se
    alinham às classes de sobrevivência.

-   O gráfico demonstra a distribuição dos passageiros por "Age" e
    "Fare", com os centróides marcados em vermelho.

------------------------------------------------------------------------

## 3. Modelo KNN (k = 5)

### 3.1 Objetivo

O **KNN** é um algoritmo supervisionado que classifica um novo ponto com
base na proximidade de seus vizinhos mais próximos.\
Neste caso, buscamos prever diretamente a variável **Survived**.

### 3.2 Resultados

-   **Acurácia**: **0.65**\

-   **Classification Report**:

        Não Sobreviveu: precision=0.68, recall=0.75, f1=0.72
        Sobreviveu:    precision=0.59, recall=0.51, f1=0.55

-   **Matriz de Confusão**:

        [[95 31]
         [44 45]]

### 3.3 Interpretação

-   O modelo tem desempenho moderado, com melhor recall para a classe
    "Não Sobreviveu".
-   O gráfico mostra a fronteira de decisão do KNN, onde áreas são
    classificadas como propensas à sobrevivência ou não.
-   A distribuição dos passageiros demonstra forte sobreposição entre
    classes, dificultando o trabalho do KNN.

------------------------------------------------------------------------

## 4. Comparação Geral entre K-Means e KNN

  Critério          K-Means              KNN
  ----------------- -------------------- -------------------------
  Tipo              Não supervisionado   Supervisionado
  Objetivo          Criar clusters       Classificar diretamente
  Acurácia obtida   0.67                 0.65
  Utilidade         Explorar padrões     Predição efetiva

### Observações

-   Mesmo sendo não supervisionado, o K-Means obteve acurácia semelhante
    ao KNN, o que demonstra forte sobreposição de padrões no dataset.
-   O KNN sofre com limites pouco definidos entre as classes, como visto
    na fronteira de decisão.

------------------------------------------------------------------------

## 5. Conclusão

Ambos os modelos apresentaram desempenho similar, mas o **KNN é mais
apropriado para a tarefa**, pois é supervisionado.\
O K-Means, apesar de não ser ideal para classificação, oferece insights
sobre agrupamentos naturais no dataset.

O experimento ilustra bem as diferenças entre algoritmos supervisionados
e não supervisionados, bem como suas limitações ao lidar com dados reais
e complexos como o Titanic.

------------------------------------------------------------------------



