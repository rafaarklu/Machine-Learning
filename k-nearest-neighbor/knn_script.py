import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler # Adicionado para escalonamento
import seaborn as sns
import pandas as pd
import kagglehub

# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DE DADOS

# O arquivo foi lido do ambiente de execução
path = kagglehub.dataset_download("adityadesai13/used-car-dataset-ford-and-mercedes")
df = pd.read_csv(path + "/bmw.csv")

# Selecionar variáveis contínuas (features)
X = df[['engineSize', 'mileage']]

# Codificar a variável alvo 'fuelType'
# fuel_labels é um objeto Index do Pandas
y, fuel_labels = pd.factorize(df['fuelType'])

# Limpeza dos dados
data = X.copy()
data['target'] = y
# O passo dropna() é mantido, embora os dados originais (mileage, engineSize) não tivessem nulos
data = data.dropna() 

X_clean = data[['engineSize', 'mileage']]
y_clean = data['target']

# 2. ESCALONAMENTO DOS DADOS (CRÍTICO PARA KNN)
scaler = StandardScaler()
# Ajustar e transformar as features
X_scaled = scaler.fit_transform(X_clean)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)


# 3. DIVISÃO TREINO/TESTE
# Usamos os dados escalonados (X_scaled_df) para o treino
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y_clean,
    test_size=0.3,
    random_state=42
)

# 4. TREINAMENTO E AVALIAÇÃO DO MODELO KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(f"Accuracy (com escalonamento): {accuracy_score(y_test, predictions):.2f}")


# 5. VISUALIZAÇÃO DA FRONTEIRA DE DECISÃO

# Reduzir a amostra para visualização (para o scatter plot)
sample_size = 5000
# Usamos o índice para garantir que X_vis e y_vis correspondam
X_vis = X_scaled_df.sample(sample_size, random_state=42)
y_vis = y_clean.loc[X_vis.index]

plt.figure(figsize=(12, 10))

# Definir os limites da grade a partir dos dados ESCALONADOS
h = 0.05 
x_min, x_max = X_scaled_df['engineSize'].min() - 0.5, X_scaled_df['engineSize'].max() + 0.5
y_min, y_max = X_scaled_df['mileage'].min() - 0.5, X_scaled_df['mileage'].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Criar a grade de pontos para predição
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Previsão KNN para cada ponto da grade
Z = knn.predict(grid_points)
Z = Z.reshape(xx.shape)

# Plot da fronteira e dos pontos
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(x=X_vis['engineSize'], y=X_vis['mileage'], hue=y_vis, style=y_vis,
                palette="deep", s=100, legend='full')

# Configurar legenda e rótulos
handles, _ = plt.gca().get_legend_handles_labels()

# CORREÇÃO DO ERRO: Converter o Index (fuel_labels) para uma lista
plt.legend(handles=handles, labels=fuel_labels.tolist(), title="Fuel Type")

plt.xlabel("Engine Size (Scaled)")
plt.ylabel("Mileage (Scaled)")
plt.title("KNN Decision Boundary (k=3) on Scaled Data")

 
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())