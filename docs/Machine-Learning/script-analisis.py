import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import StringIO
import kagglehub

# Baixar e carregar o dataset BMW
path = kagglehub.dataset_download("adityadesai13/used-car-dataset-ford-and-mercedes")
df = pd.read_csv(path + "/bmw.csv")

# Selecionar variáveis para análise
x = df['engineSize'].values
y = df['mpg'].values

# Remover valores ausentes
mask = ~np.isnan(x) & ~np.isnan(y)
x = x[mask]
y = y[mask]

plt.rcParams["figure.figsize"] = (5, 3)  

xa = 2.0  # exemplo de engineSize
ya = np.interp(xa, x, y)  # valor aproximado de mpg para engineSize=xa
k = 0.3
ka = xa - k
ak = xa + k

fig, ax = plt.subplots(1, 1)
ax.axhline(0, color='gray') # x = 0
ax.axvline(0, color='gray') # y = 0
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.scatter(x, y, color='red', alpha=0.3, label='engineSize vs mpg')
ax.set_xlim(min(x)-0.2, max(x)+0.2)
ax.set_xticks([])
ax.set_yticks([])
ax.plot([ka, ka], [0, np.interp(ka, x, y)], 'g:')
ax.plot([0, ka], [np.interp(ka, x, y), np.interp(ka, x, y)], 'g:')
ax.plot([ak, ak], [0, np.interp(ak, x, y)], 'g:')
ax.plot([0, ak], [np.interp(ak, x, y), np.interp(ak, x, y)], 'g:')
ax.text(xa, -2, 'a', horizontalalignment='center', fontsize=15)
ax.text(ka, -2, '$a-\delta$', horizontalalignment='center', fontsize=15)
ax.text(ak, -2, '$a+\delta$', horizontalalignment='center', fontsize=15)
ax.text(0, np.interp(ka, x, y), '$L-\epsilon$', horizontalalignment='right', verticalalignment='center', fontsize=15)
ax.text(0, np.interp(ak, x, y), '$L+\epsilon$', horizontalalignment='right', verticalalignment='center', fontsize=15)
ax.plot([xa, xa], [0, np.interp(xa, x, y)], 'b:')
ax.plot([0, xa], [np.interp(xa, x, y), np.interp(xa, x, y)], 'b:')
ax.plot(xa, np.interp(xa, x, y), 'ro', ms=15)
ax.text(0, np.interp(xa, x, y), 'L=f(a)', horizontalalignment='right', verticalalignment='center', fontsize=15)

fig.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())