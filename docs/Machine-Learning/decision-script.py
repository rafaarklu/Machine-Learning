import matplotlib.pyplot as plt
import pandas as pd

import kagglehub
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

plt.figure(figsize=(12, 10))


path = kagglehub.dataset_download("adityadesai13/used-car-dataset-ford-and-mercedes")

df = pd.read_csv(path + "/bmw.csv")  # Adjust filename as needed
x = df[['model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'tax', 'engineSize']]

label_encoder = LabelEncoder()



# Carregar o conjunto de dados
label_encoder = LabelEncoder()
x['model'] = label_encoder.fit_transform(x['model'])
x['transmission'] = label_encoder.fit_transform(x['transmission'])
x['fuelType'] = label_encoder.fit_transform(x['fuelType'])  

#setar a saida
y= df['consumo_cat'] = pd.cut(
        df['mpg'],
        bins=[0, 25, 40, 100],   # faixas (ajustáveis)
        labels=['baixo', 'medio', 'alto']
)



# After creating x and y
data = x.copy()
data['target'] = y

# Drop rows with NaN in any column
data = data.dropna()

# Split features and target again
x_clean = data.drop('target', axis=1)
y_clean = data['target']

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x_clean, y_clean, 
    test_size=0.2, 
    random_state=42
)


# Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Avaliar o modelo
accuracy = classifier.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
tree.plot_tree(classifier)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())