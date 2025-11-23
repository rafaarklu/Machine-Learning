import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("Titanic-Dataset.csv")
# A maioria dos bilhetes é única, então mostramos o Top 10
data_counts = df["Ticket"].value_counts().nlargest(10)

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    data_counts.index, 
    data_counts.values, 
    color="#17becf", 
    edgecolor="black", 
    alpha=0.9
)

ax.set_title("Contagem dos Top 10 Números de Bilhete", fontsize=16, pad=15)
ax.set_xlabel("Número do Bilhete", fontsize=12)
ax.set_ylabel("Contagem", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.5, linestyle='--')

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())