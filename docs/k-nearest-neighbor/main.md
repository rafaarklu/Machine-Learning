# Machine Learning com KNN 

## Tabela utilizada

[https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes/data](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes/data)

-------------------------------------------------------------------------------------------------------------------------


Assim como na Arvore de Decisão, utilizarei a mesma base de dados de carros usados da bmw.
PORÉM, desta vez irei prever o tipo de combustivel utilizando os seguintes valores:

+ Model (modelo) 
+ Year (ano do carro)
+ Transmission (tipo de transmissão do carro)
+ Engine Size (tamanho do motor do carro)

-------------------------------------------------------------------------------------------------------------------------

## Modelo KNN

=== "output"

    ``` python exec="on" html="1"
    --8<-- "./docs/k-nearest-neighbor/knn_script.py"
    ```

=== "code"

    ``` python exec="off"
    --8<-- "./docs/k-nearest-neighbor/knn_script.py"
    ```


-------------------------------------------------------------------------------------------------------------------------

## Como Fazer?


* Baixar e importar bibliotecas

```python exec="off"
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import kagglehub
import pandas as pd
from sklearn.preprocessing import LabelEncoder
```

* "Setar" o caminho de onde está a base de dados a ser utilizada (Neste exemplo usei uma base retirada do Keggle, porém existem infinitas outras formas de trazer bases de dados)

```python exec= "off"
plt.figure(figsize=(12, 10))


path = kagglehub.dataset_download("adityadesai13/used-car-dataset-ford-and-mercedes")

df = pd.read_csv(path + "/bmw.csv")  
x = df[['model', 'year', 'transmission', 'engineSize',]]
```

* Transformar as variáveis que serão utilizadas para numerais (caso sejam strings, que é o caso aqui)

```python exec= "off"

label_encoder = LabelEncoder()
x['model'] = label_encoder.fit_transform(x['model'])
x['year'] = label_encoder.fit_transform(x['year'])  
x['transmission'] = label_encoder.fit_transform(x['transmission'])  
x['engineSize'] = label_encoder.fit_transform(x['engineSize'])


y = LabelEncoder().fit_transform(df['fuelType'])

data = x.copy()
data['target'] = y


```

* Limpar as linhas que possuem valores ausentes

``` python exec= "off"
data = data.dropna()

x_clean = data.drop('target', axis=1)
y_clean = data['target']

```

* Treinar o modelo

```pyton exec= "off"
x_train, x_test, y_train, y_test = train_test_split(
    x_clean, y_clean, 
    test_size=0.7, 
    random_state=42
)
 
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")


```

* 