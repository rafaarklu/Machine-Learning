import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import kagglehub
import os
from io import StringIO
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix



path = kagglehub.dataset_download("yasserh/titanic-dataset")
file_path = os.path.join(path, "Titanic-Dataset.csv")
df = pd.read_csv(file_path)



features = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
labels_true = df['Survived'].copy()

# Preprocessamento
features['Sex'] = LabelEncoder().fit_transform(features['Sex'])
features['Age'].fillna(features['Age'].median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)



# K-Means
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100, random_state=42, n_init=10)
labels_pred = kmeans.fit_predict(X_scaled)

# Ajuste de rótulo (clusters podem estar invertidos)
if accuracy_score(labels_true, labels_pred) < accuracy_score(labels_true, 1-labels_pred):
    labels_pred = 1-labels_pred

# Avaliação
acc = accuracy_score(labels_true, labels_pred)
cm = confusion_matrix(labels_true, labels_pred)
print(f"Acurácia do K-Means para prever 'Survived': {acc:.2f}")
print("Matriz de confusão:")
print(cm)

# Visualização dos clusters
x_var = 'Age'
y_var = 'Fare'
x_idx = features.columns.get_loc(x_var)
y_idx = features.columns.get_loc(y_var)

plt.figure(figsize=(10, 8))
plt.scatter(X_scaled[:, x_idx], X_scaled[:, y_idx], c=labels_pred, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, x_idx], kmeans.cluster_centers_[:, y_idx],
            c='red', marker='*', s=200, label='Centroids')
plt.title("K-Means Clustering - Titanic Dataset")
plt.xlabel(f"{x_var} (scaled)")
plt.ylabel(f"{y_var} (scaled)")
plt.legend()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())