import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import kagglehub
import os

# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DE DADOS

# Baixar o dataset Titanic via kagglehub
path = kagglehub.dataset_download("yasserh/titanic-dataset")
file_path = os.path.join(path, "Titanic-Dataset.csv")
df = pd.read_csv(file_path)

# Selecionar features numéricas: Age, Fare, Pclass
# Variável alvo binária: Survived (0 = não sobreviveu, 1 = sobreviveu)
X = df[['Pclass', 'Age', 'Fare']].copy()

# Variável alvo binária
y = df['Survived'].copy()

# Limpeza dos dados (remover linhas com valores faltantes)
data = X.copy()
data['Survived'] = y
data = data.dropna()

X_clean = data[['Pclass', 'Age', 'Fare']]
y_clean = data['Survived']

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
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# Métricas de desempenho
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy (KNN com k=5): {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=['Não Sobreviveu', 'Sobreviveu']))

# Matriz de confusão
cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)


# 5. VISUALIZAÇÃO DA FRONTEIRA DE DECISÃO

# Para visualização em 2D, vamos usar apenas Age e Fare
X_2d = X_scaled_df[['Age', 'Fare']].copy()
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y_clean,
    test_size=0.3,
    random_state=42
)

# Treinar KNN com os dados 2D
knn_2d = KNeighborsClassifier(n_neighbors=5)
knn_2d.fit(X_train_2d, y_train_2d)

# Reduzir a amostra para visualização (para o scatter plot)
sample_size = min(1000, len(X_2d))
# Usamos o índice para garantir que X_vis e y_vis correspondam
X_vis = X_2d.sample(sample_size, random_state=42)
y_vis = y_clean.loc[X_vis.index]

plt.figure(figsize=(12, 8))

# Definir os limites da grade a partir dos dados ESCALONADOS
h = 0.05 
x_min, x_max = X_2d['Age'].min() - 0.5, X_2d['Age'].max() + 0.5
y_min, y_max = X_2d['Fare'].min() - 0.5, X_2d['Fare'].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Criar a grade de pontos para predição
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Previsão KNN para cada ponto da grade
Z = knn_2d.predict(grid_points)
Z = Z.reshape(xx.shape)

# Plot da fronteira e dos pontos
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(x=X_vis['Age'], y=X_vis['Fare'], hue=y_vis, style=y_vis,
                palette="deep", s=100, legend='full')

# Configurar legenda e rótulos
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=['Não Sobreviveu', 'Sobreviveu'], title="Titanic - Survivability")

plt.xlabel("Age (Scaled)")
plt.ylabel("Fare (Scaled)")
plt.title("KNN Decision Boundary (k=5) - Titanic Dataset")

 
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())