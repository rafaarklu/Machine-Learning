import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import numpy as np

df = pd.read_csv("bmw.csv")
data = df["year"].dropna()
bins = int(np.sqrt(len(data)))

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(data, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.9)

ax.set_title("Distribuição do Ano de Fabricação dos Carros BMW", fontsize=16, pad=15)
ax.set_xlabel("Ano de Fabricação", fontsize=12)
ax.set_ylabel("Frequência", fontsize=12)
plt.grid(axis='y', alpha=0.5, linestyle='--')
plt.ticklabel_format(style='plain', axis='x') 

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())