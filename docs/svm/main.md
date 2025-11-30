# O dataset
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
# Support Vector Machine (SVM)

Support Vector Machine

O Support Vector Machine (SVM) é um algoritmo supervisionado amplamente utilizado para classificação (e também disponível em variantes de regressão, como SVR). O objetivo do SVM é encontrar a melhor fronteira (linha, plano ou hiperplano) que separa classes no espaço de features.

## Conceito básico

Em problemas de classificação binária (por exemplo, "sobreviveu" vs "não sobreviveu"), o SVM busca o hiperplano que maximize a margem — ou seja, a distância entre o hiperplano e os pontos mais próximos de cada classe (os vetores de suporte).

Quanto maior a margem, mais robusta tende a ser a separação entre classes.

## Intuição visual

Imagine duas nuvens de pontos. Embora várias linhas possam separar as nuvens, o SVM seleciona a que:

- maximiza a margem;
- está o mais distante possível dos pontos das duas classes;
- é definida por poucos pontos críticos (os vetores de suporte).

## Kernel Trick

Quando os dados não são linearmente separáveis no espaço original, o SVM pode aplicar uma transformação (kernel) que projeta os dados para um espaço de maior dimensão onde a separação é possível.

### Principais kernels

- **Linear**: separação por uma linha/hiperplano;
- **Polinomial**: permite curvas polinomiais (grau 2, 3, ...);
- **RBF (Radial Basis Function)**: produz fronteiras complexas; é o mais usado na prática;
- **Sigmoid**: comportamento similar a redes neurais simples.


# Script e resultado

=== "output"

    ``` python exec="on" html="1"
    --8<-- "./docs/svm/svm.py"
    ```

=== "code"

    ``` python exec="off"
    --8<-- "./docs/svm/svm.py"
    ```

## O script explicado

Esta seção descreve passo a passo o script utilizado no exemplo (arquivo `docs/svm/svm.py`). Trechos de código relevantes foram convertidos em blocos Python para facilitar a leitura e a execução em MkDocs.

### 1. Carregamento e preparação dos dados

```python
df = pd.read_csv("Titanic-Dataset.csv")
df = df[['Survived', 'Age', 'Fare', 'Sex']]
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = df.dropna()
```

- Carrega o dataset do Titanic;
- Seleciona variáveis relevantes;
- Converte `Sex` para valores numéricos;
- Remove valores faltantes (necessário para treinar o modelo).

### 2. Seleção de features para visualização

```python
X = df[['Age', 'Fare']].values
y = df['Survived'].values
```

Nesta demonstração usamos apenas `Age` e `Fare` para manter os dados em 2D (necessário para plotar as fronteiras de decisão com `DecisionBoundaryDisplay`).

### 3. Padronização

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

O SVM costuma se beneficiar de dados escalonados, especialmente para kernels como RBF e polinomial.

### 4. Treinamento com múltiplos kernels

```python
kernels = {
    'linear': ax1,
    'sigmoid': ax2,
    'poly': ax3,
    'rbf': ax4
}

for k, ax in kernels.items():
    svm = SVC(kernel=k, C=1)
    svm.fit(X, y)
```

O script treina um modelo SVM para cada kernel (`linear`, `sigmoid`, `poly`, `rbf`) para comparar as fronteiras de decisão.

### 5. Plotando a fronteira de decisão

```python
DecisionBoundaryDisplay.from_estimator(
    svm,
    X,
    response_method="predict",
    alpha=0.8,
    cmap="Pastel1",
    ax=ax
)
```

Essa função desenha a fronteira de decisão aprendida pelo SVM. A forma da fronteira varia conforme o kernel:

- linear: linha reta;
- poly: curvas suaves;
- rbf: fronteiras detalhadas e não lineares;
- sigmoid: separações parecidas com funções tipo rede neural.

### 6. Plotando os pontos de dados

```python
ax.scatter(
    X[:, 0], X[:, 1],
    c=y,
    s=20, edgecolors="k"
)
```

Isso plota os pontos reais do dataset sobre a superfície de decisão (1 = sobrevivente, 0 = não sobrevivente).

### 7. Salvando a figura como SVG em buffer

```python
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
```

Em vez de mostrar a imagem interativamente, o script pode retornar o conteúdo SVG — útil para inclusão em páginas HTML ou MkDocs.

## O que a visualização permite analisar

- Regiões onde o modelo prevê sobrevivência vs. não sobrevivência;
- Complexidade das fronteiras de decisão e flexibilidade do modelo;
- Comparação direta entre kernels (rigidez vs. flexibilidade).

Exemplos:

- Linear: fronteira reta;
- Poly: curvas mais suaves;
- RBF: fronteiras com muitos detalhes;
- Sigmoid: comportamento intermediário.

## Conclusão

O SVM é um modelo poderoso que:

- Busca a melhor separação entre classes;
- Utiliza vetores de suporte como pontos críticos;
- Pode gerar fronteiras lineares ou altamente não lineares via kernels;
- Se beneficia de escalonamento dos dados.

O exemplo prático demonstra como diferentes kernels afetam a decisão do modelo no problema do Titanic e ajuda a entender seu comportamento.

---

