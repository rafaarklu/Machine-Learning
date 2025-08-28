# Machine Learning com Árvore de Decisão

## Tabela utilizada

[https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes/data](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes/data)

-------------------------------------------------------------------------------------------------

Essa Arquivo .CSV possui dados de carros usados da marca BWM nos Estados Unidos, contendo as seguintes metricas:

'model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'tax', 'engineSize'

Com esses dados é possível montar um modelo preditivo para descobrir o gasto de combustivel ('Alto, médio ou baixo')

Obs: medida utilizada é mpg (miles per galon)

-------------------------------------------------------------------------------------------------

## Tabela de Treino

``` python exec="on" html="1"
--8<-- "./docs/Machine-Learning/decision-script.py"
```

-------------------------------------------------------------------------------------------------

## Passo a passo do script de Árvore de Decisão

### 1. Importação de Bibliotecas
O script importa bibliotecas essenciais para manipulação de dados (`pandas`), visualização (`matplotlib`), acesso ao Kaggle (`kagglehub`), e machine learning (`sklearn`).

### 2. Download e Carregamento dos Dados
Utiliza o `kagglehub` para baixar o dataset e carrega o arquivo `bmw.csv` em um DataFrame do pandas.

### 3. Seleção de Features
Seleciona as colunas relevantes para o modelo: `'model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'tax', 'engineSize'`.

### 4. Codificação de Variáveis Categóricas
Utiliza o `LabelEncoder` para transformar variáveis categóricas (`model`, `transmission`, `fuelType`) em valores numéricos.

### 5. Criação da Variável Alvo
A variável alvo (`consumo_cat`) é criada a partir da coluna `mpg`, categorizando o consumo em `'baixo'`, `'medio'` e `'alto'` usando `pd.cut`.

### 6. Limpeza dos Dados
Remove linhas com valores ausentes para garantir que o modelo seja treinado apenas com dados válidos.

### 7. Separação em Treino e Teste
Divide os dados em conjuntos de treino e teste, usando 70% dos dados para teste (`test_size=0.7`).

### 8. Treinamento do Modelo
Cria e treina um classificador de árvore de decisão (`DecisionTreeClassifier`) com os dados de treino.

### 9. Avaliação do Modelo
Calcula a acurácia do modelo no conjunto de teste e exibe o valor.

### 10. Visualização da Árvore
Plota a árvore de decisão treinada usando `matplotlib`.

### 11. Exportação do Gráfico
Salva o gráfico da árvore em formato SVG para visualização em HTML.

-------------------------------------------------------------------------------------------------

## Observações

- O modelo prevê a categoria de consumo de combustível com base nas características do carro.
- É possível ajustar os intervalos de consumo (`bins`) conforme necessário.
- O script pode ser adaptado para outros datasets ou