import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
import kagglehub

# Carregar o dataset da BMW via KaggleHub
path = kagglehub.dataset_download("adityadesai13/used-car-dataset-ford-and-mercedes")
df = pd.read_csv(path + "/bmw.csv")  

# Selecionar variáveis contínuas para o gráfico 2D
x = df[['engineSize', 'mileage']]

# Codificar a variável alvo 'fuelType'
y, fuel_labels = pd.factorize(df['fuelType'])

# Preparar os dados
data = x.copy()
data['target'] = y
data = data.dropna()

x_clean = data.drop('target', axis=1)
y_clean = data['target']

# Divisão treino/teste
x_train, x_test, y_train, y_test = train_test_split(
    x_clean, y_clean, 
    test_size=0.3, 
    random_state=42
)

# Treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# Reduzir a amostra para visualização (para não estourar memória)
sample_size = 5000
x_vis = x_clean.sample(sample_size, random_state=42)
y_vis = y_clean.loc[x_vis.index]

# Visualizar o limite de decisão
h = 50  # passo maior para reduzir o tamanho da grade
x_min, x_max = x_vis['engineSize'].min() - 1, x_vis['engineSize'].max() + 1
y_min, y_max = x_vis['mileage'].min() - 1000, x_vis['mileage'].max() + 1000

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Previsão KNN para cada ponto da grade
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array(Z)  # garante 1D
Z = Z.reshape(xx.shape)  # reshape para o tamanho da grade

# Plot
plt.figure(figsize=(12, 10))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(x=x_vis['engineSize'], y=x_vis['mileage'], hue=y_vis, style=y_vis,
                palette="deep", s=100, legend='full')

handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=fuel_labels, title="Fuel Type")
# Exibir o gráfico
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
