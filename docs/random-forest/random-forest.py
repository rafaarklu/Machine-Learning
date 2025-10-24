from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import kagglehub
import pandas as pd
import os

# 1. Baixar e carregar os dados
path = kagglehub.dataset_download("yasserh/titanic-dataset")
file_path = os.path.join(path, "Titanic-Dataset.csv")
df = pd.read_csv(file_path)

# 2. Selecionar colunas relevantes
features = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']].copy()

# 3. Tratar valores ausentes
features['Age'].fillna(features['Age'].mean(), inplace=True)
features['Embarked'].fillna(features['Embarked'].mode()[0], inplace=True)

# 4. Converter variáveis categóricas em numéricas (One-Hot Encoding)
features = pd.get_dummies(features, columns=['Sex', 'Embarked'], drop_first=True)

# 5. Definir X (entradas) e y (alvo)
X = features.drop('Survived', axis=1)
y = features['Survived']

# 6. Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Criar modelo Random Forest
rf = RandomForestClassifier(
    n_estimators=100,        # número de árvores
    max_depth=None,          # profundidade ilimitada
    max_features='sqrt',     # nº de features consideradas em cada split
    oob_score=True,          # validação out-of-bag
    random_state=42
)

# 8. Treinar modelo
rf.fit(X_train, y_train)

# 9. Avaliar modelo
y_pred = rf.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("OOB Score:", rf.oob_score_)
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# 10. Importância das variáveis
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nImportância das variáveis:\n", importances)
