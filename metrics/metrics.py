import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO # Importação para a saída de gráfico em SVG
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import kagglehub
import os

# 1. EXPLORAÇÃO E PREPARAÇÃO DOS DADOS
# Carrega o arquivo 'Titanic-Dataset.csv'
path = kagglehub.dataset_download("yasserh/titanic-dataset")
file_path = os.path.join(path, "Titanic-Dataset.csv")
df = pd.read_csv(file_path)

# Definição das Features e do Target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

X = df[features]
y = df[target]

# 2. PRÉ-PROCESSAMENTO AVANÇADO (Pipeline)
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_features = ['Pclass', 'Sex', 'Embarked'] 

# Pipeline para features Numéricas: Imputação (mediana) + Escalonamento (StandardScaler)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) # CRUCIAL para KNN
])

# Pipeline para features Categóricas: Imputação (mais frequente) + One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer combina os pipelines de pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# 3. DIVISÃO DOS DADOS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TREINAMENTO DO MODELO (KNN)
# Pipeline completo: Pré-processador + Classificador KNN (k=5)
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5)) 
])

knn_pipeline.fit(X_train, y_train)

# 5. AVALIAÇÃO DO MODELO
y_pred = knn_pipeline.predict(X_test)

# Métricas de Desempenho
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Acurácia (Accuracy): {accuracy:.4f}")
print("\nMatriz de Confusão:")
print(conf_matrix)
print("\nRelatório de Classificação (Precision, Recall, F1-Score):")
print(class_report)

# Gera o gráfico da Matriz de Confusão
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Não Sobreviveu (0)', 'Sobreviveu (1)'],
            yticklabels=['Não Sobreviveu (0)', 'Sobreviveu (1)'])
plt.ylabel('Valor Verdadeiro')
plt.xlabel('Predição')
plt.title('Matriz de Confusão do KNN (k=5)')

# SALVA O GRÁFICO NO BUFFER E IMPRIME O CONTEÚDO SVG
buffer = BytesIO()
plt.savefig(buffer, format="svg", transparent=True)
buffer.seek(0)
print(buffer.getvalue().decode("utf-8"))