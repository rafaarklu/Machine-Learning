import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("Titanic-Dataset.csv")
data_counts = df["Sex"].value_counts()

fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(
    data_counts.index, 
    data_counts.values, 
    color=["#d62728", "#1f77b4"], 
    edgecolor="black", 
    alpha=0.9
)

ax.set_title("Contagem por Gênero", fontsize=16, pad=15)
ax.set_xlabel("Gênero", fontsize=12)
ax.set_ylabel("Contagem", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.5, linestyle='--')

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())