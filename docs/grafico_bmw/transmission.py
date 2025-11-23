import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("bmw.csv")
# Conta a frequência de cada tipo de transmissão
data_counts = df["transmission"].value_counts()

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    data_counts.index, 
    data_counts.values, 
    color="#ff7f0e", # Laranja
    edgecolor="black", 
    alpha=0.9
)

ax.set_title("Frequência dos Tipos de Transmissão", fontsize=16, pad=15)
ax.set_xlabel("Tipo de Transmissão", fontsize=12)
ax.set_ylabel("Contagem", fontsize=12)
plt.xticks(rotation=0) 
plt.grid(axis='y', alpha=0.5, linestyle='--')

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())