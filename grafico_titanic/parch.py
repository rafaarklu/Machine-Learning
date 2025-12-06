import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("Titanic-Dataset.csv")
data_counts = df["Parch"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    data_counts.index.astype(str),
    data_counts.values, 
    color="#8c564b", 
    edgecolor="black", 
    alpha=0.9
)

ax.set_title("Contagem de Pais/Crianças (Parch)", fontsize=16, pad=15)
ax.set_xlabel("Número de Pais/Crianças", fontsize=12)
ax.set_ylabel("Contagem", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.5, linestyle='--')

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())