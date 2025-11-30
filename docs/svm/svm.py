import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from io import StringIO
from sklearn.preprocessing import StandardScaler

# ============================
# Carregar dataset
# ============================
df = pd.read_csv("Titanic-Dataset.csv")

# Selecionar apenas as colunas necessárias
df = df[['Survived', 'Age', 'Fare', 'Sex']]

# Converter variável categórica
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Remover valores faltantes
df = df.dropna()

# ============================
# Seleção de Features para plot 2D
# ============================
# Usaremos apenas Age e Fare para plotar a fronteira
X = df[['Age', 'Fare']].values
y = df['Survived'].values

# Padronizar
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ============================
# Plot
# ============================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))

kernels = {
    'linear': ax1,
    'sigmoid': ax2,
    'poly': ax3,
    'rbf': ax4
}

for k, ax in kernels.items():
    svm = SVC(kernel=k, C=1)
    svm.fit(X, y)

    DecisionBoundaryDisplay.from_estimator(
        svm,
        X,
        response_method="predict",
        alpha=0.8,
        cmap="Pastel1",
        ax=ax
    )

    ax.scatter(
        X[:, 0], X[:, 1],
        c=y,
        s=20,
        edgecolors="k"
    )

    ax.set_title(f"SVM Kernel: {k}")
    ax.set_xticks([])
    ax.set_yticks([])

# Salvar como SVG em buffer
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()
