

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import kagglehub
import os
from io import BytesIO



path = kagglehub.dataset_download("yasserh/titanic-dataset")
file_path = os.path.join(path, "Titanic-Dataset.csv")
df = pd.read_csv(file_path)



features = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()

features['Sex'] = LabelEncoder().fit_transform(features['Sex'])

features['Age'].fillna(features['Age'].median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

x_var = 'Age'
y_var = 'Fare'
x_idx = features.columns.get_loc(x_var)
y_idx = features.columns.get_loc(y_var)


kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)



plt.figure(figsize=(10, 8))

plt.scatter(X_scaled[:, x_idx], X_scaled[:, y_idx], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, x_idx], kmeans.cluster_centers_[:, y_idx],
            c='red', marker='*', s=200, label='Centroids')

plt.title("K-Means Clustering - Titanic Dataset")
plt.xlabel(f"{x_var} (scaled)")
plt.ylabel(f"{y_var} (scaled)")
plt.legend()


buffer = BytesIO()
plt.savefig(buffer, format="svg", transparent=True)
buffer.seek(0)
print(buffer.getvalue().decode("utf-8"))
