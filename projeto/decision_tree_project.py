import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import kagglehub

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from math import sqrt

# ----------------------------
# 1. Baixar dataset do Kaggle
# ----------------------------
path = kagglehub.dataset_download("wardabilal/salary-prediction-dataset")
df = pd.read_csv(path + "/Salary_Data.csv")

# ----------------------------
# 2. Pré-processamento
# ----------------------------
df = df.dropna(subset=['Salary']).copy()

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Years of Experience'] = df['Years of Experience'].fillna(df['Years of Experience'].median())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode().iloc[0])
df['Education Level'] = df['Education Level'].fillna(df['Education Level'].mode().iloc[0])
df['Job Title'] = df['Job Title'].fillna(df['Job Title'].mode().iloc[0])

X = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
y = df['Salary']

for col in ['Gender', 'Education Level', 'Job Title']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# ----------------------------
# 3. Treino e teste
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

regressor = DecisionTreeRegressor(
    random_state=42,
    max_depth=5,
    min_samples_leaf=5
)
regressor.fit(X_train, y_train)

# ----------------------------
# 4. Avaliação
# ----------------------------
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

# ----------------------------
# 5. Visualização (para MkDocs)
# ----------------------------
plt.figure(figsize=(20, 12))
plot_tree(
    regressor,
    feature_names=X.columns,
    filled=True,
    fontsize=8,
    max_depth=3   # limita profundidade do desenho para ficar legível
)

buffer = StringIO()
plt.savefig(buffer, format="svg", bbox_inches="tight")
print(buffer.getvalue())
