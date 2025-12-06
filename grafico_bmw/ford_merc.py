import pandas as pd
import kagglehub
 


import pandas as pd
path = kagglehub.dataset_download("adityadesai13/used-car-dataset-ford-and-mercedes")
df = pd.read_csv(path + "/bmw.csv")  


# Exibe apenas 10 linhas aleatórias em formato markdown
print(df.sample(n=15).to_markdown(index=False))