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

 

# Objetivo do projeto

Aplicar o algoritmo **K-Nearest Neighbors (KNN)** para um problema de **classificação binária**: prever a variável `Survived` do dataset Titanic.  
Para facilitar a visualização da fronteira de decisão, o modelo usa apenas duas features contínuas: **Age** e **Fare**.

---

# Intuição do KNN

KNN é um método **baseado em instâncias** (lazy learning). Para classificar um ponto novo, o algoritmo encontra os `k` pontos mais próximos no conjunto de treino (vizinhos) e decide a classe pela **maioria** entre esses vizinhos (classificação) ou pela **média** (regressão).  
A premissa é que pontos próximos no espaço de features tendem a ter o mesmo rótulo.

---

# Por que escalonar (StandardScaler) é crítico

KNN usa distâncias; se duas features têm escalas muito diferentes (ex.: `Fare` em centenas vs `Age` em dezenas), a feature com maior escala dominará a distância e irá enviesar a decisão.  
O `StandardScaler` transforma cada feature de modo que todas tenham média 0 e desvio padrão 1, tornando as features comparáveis.

---

# Complexidade e limitações práticas

- **Complexidade na predição**: O(n · d) por instância (n = nº de amostras de treino, d = dimensão). Para muitos pontos de treino, predição fica cara.  
- **Curse of dimensionality**: em alta dimensão, distâncias tornam-se menos discriminativas → KNN perde eficácia.  
- **Sensível a ruído e outliers**: vizinhos ruidosos mudam a predição.  
- **Escolha de k**: pequeno → mais variance (overfitting); grande → mais bias (underfitting).  
- **Balanceamento de classes**: em caso de classes desbalanceadas, vizinhança pode ser dominada pela classe majoritária local.

Acelerações: KD-Tree, Ball-Tree, aproximações (approx. nearest neighbors) ou redução de dimensionalidade (PCA, UMAP).

---

# Hiperparâmetros importantes e tuning

- `n_neighbors (k)`: testar via validação cruzada (CV). Valores comuns: 3,5,7,9.  
- `metric`: distância a usar (euclidiana, manhattan, minkowski).  
- `weights`: `uniform` (voto igual) ou `distance` (vizinhos mais próximos têm mais peso).  
- Pré-processamento: escolha de scaler, tratamento de outliers, e engenharia de features.

Dica prática: use `GridSearchCV` ou `cross_val_score` para escolher `k` com base em métricas (accuracy, F1, ROC-AUC).

---



# Código Exemplo e gráfico

=== "output"

    ``` python exec="on" html="1"
    --8<-- "./docs/k-nearest-neighbor/knn_script.py"
    ```

=== "code"

    ``` python exec="off"
    --8<-- "./docs/k-nearest-neighbor/knn_script.py"
    ```

---

# Interpretação dos resultados

- **Accuracy e classification report** dão uma visão global do desempenho (precision, recall, F1) por classe.  
- **Fronteira de decisão**: ilustra como o modelo divide o espaço `Age × Fare`.  
- **Insights práticos**: passageiro com `Fare` escalado alto tende a ser classificado como `Sobreviveu` (reflexo de classes sociais/posicionamento no navio). Padrões por idade também aparecem (ex.: grupos jovens ou muito idosos podem ter maior ou menor probabilidade, dependendo dos vizinhos).

Lembre-se: com apenas duas features, o modelo captura apenas parte da realidade — usar mais features melhora (ou complica) o que o KNN enxerga.

---

# Boas práticas

- **Escolha de k**: avaliar via validação cruzada (ex.: testar k em [1,3,5,7,9,11]).  
- **Weights**: testar `weights='distance'` para dar mais peso a vizinhos mais próximos.  
- **Métricas**: além de accuracy, usar ROC-AUC (quando adequado), F1 (em classes desbalanceadas).  
- **Balanceamento**: em datasets com classes desbalanceadas, considerar undersampling/oversampling ou métricas balanceadas.  
- **Redução de dimensionalidade**: se aumentar features, usar PCA/TSNE para visualização e possivelmente acelerar.  
- **Validação por bootstrap ou k-fold**: para estimar variabilidade da métrica.  
- **Aceleração**: kd_tree/ball_tree ou Approx Nearest Neighbors para muitos exemplos.

---


# Conclusão

KNN é simples, interpretável e poderoso em espaços de baixa dimensão com muitos exemplos. No entanto, não escala bem para grandes bases e perde desempenho em alta dimensão. No contexto do Titanic com `Age` e `Fare`, KNN oferece uma boa forma de visualizar como idade e tarifa se relacionam com a sobrevivência, mas deve ser complementado com experimentos (mais features, tuning de k e validação) para tirar conclusões robustas.

---

