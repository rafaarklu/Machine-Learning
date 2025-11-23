import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("Titanic-Dataset.csv")
data_counts = df["Survived"].value_counts().sort_index()
labels = ['Não Sobreviveu (0)', 'Sobreviveu (1)']

fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(
    labels,
    data_counts.values, 
    color=["#ff7f0e", "#2ca02c"], 
    edgecolor="black", 
    alpha=0.9
)

ax.set_title("Contagem de Sobreviventes", fontsize=16, pad=15)
ax.set_xlabel("Sobrevivência", fontsize=12)
ax.set_ylabel("Contagem", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.5, linestyle='--')

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())