import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("bmw.csv")
# Conta a frequência dos Top 10 modelos para melhor visualização
data_counts = df["model"].value_counts().nlargest(10)

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    data_counts.index, 
    data_counts.values, 
    color="#ff7f0e", # Laranja
    edgecolor="black", 
    alpha=0.9
)

ax.set_title("Contagem dos Top 10 Modelos BMW", fontsize=16, pad=15)
ax.set_xlabel("Modelo", fontsize=12)
ax.set_ylabel("Contagem", fontsize=12)
plt.xticks(rotation=45, ha='right') # Rotação para rótulos mais longos
plt.grid(axis='y', alpha=0.5, linestyle='--')

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())