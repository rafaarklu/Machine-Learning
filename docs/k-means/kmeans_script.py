# =============================
# K-Means Clustering no Titanic
# =============================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import kagglehub
import os
from io import StringIO


# ======================================
# 1. Baixar dataset do Kaggle (Titanic)
# ======================================

# Usando kagglehub para baixar
path = kagglehub.dataset_download("yasserh/titanic-dataset")

# O arquivo baixado fica dentro de path
file_path = os.path.join(path, "Titanic-Dataset.csv")

# Carregar CSV
df = pd.read_csv(file_path)

# ======================================
# 2. Preparar os dados
# ======================================

# Selecionar variáveis relevantes
features = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()

# Converter "Sex" em numérico
features['Sex'] = LabelEncoder().fit_transform(features['Sex'])

# Preencher valores ausentes
features['Age'].fillna(features['Age'].median(), inplace=True)

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# ======================================
# 3. Rodar K-Means
# ======================================
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# ======================================
# 4. Visualizar resultados
# ======================================

plt.figure(figsize=(10, 8))

# Usando só as duas primeiras features normalizadas para visualização
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='*', s=200, label='Centroids')

plt.title("K-Means Clustering - Titanic Dataset")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.legend()

# ======================================
# 5. Mostrar gráfico no MkDocs
# ======================================
from io import StringIO
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

