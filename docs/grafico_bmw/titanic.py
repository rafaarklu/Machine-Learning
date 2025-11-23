import pandas as pd
import kagglehub
import os
import pandas as pd



path = kagglehub.dataset_download("yasserh/titanic-dataset")
file_path = os.path.join(path, "Titanic-Dataset.csv")
df = pd.read_csv(file_path)



# Exibe apenas 10 linhas aleatórias em formato markdown
print(df.sample(n=15).to_markdown(index=False))