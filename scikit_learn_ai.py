# Importa a biblioteca pandas para manipulação de dados em formato de tabelas
import pandas as pd

# Carrega o arquivo CSV contendo os dados dos clientes
tabela = pd.read_csv("clientes.csv")
# Exibe informações sobre a estrutura da tabela, como tipos de dados e valores nulos
print(tabela.info())

# Importa o LabelEncoder da biblioteca scikit-learn para codificação de variáveis categóricas
from sklearn.preprocessing import LabelEncoder

# Inicializa o codificador
codificador = LabelEncoder()
# Converte a coluna "profissao" de valores categóricos para valores numéricos
tabela["profissao"] = codificador.fit_transform(tabela["profissao"])
# Converte a coluna "mix_credito" de valores categóricos para valores numéricos
tabela["mix_credito"] = codificador.fit_transform(tabela["mix_credito"])
# Converte a coluna "comportamento_pagamento" de valores categóricos para valores numéricos
tabela["comportamento_pagamento"] = codificador.fit_transform(tabela["comportamento_pagamento"])
# Exibe as informações da tabela atualizada
print(tabela.info())

# Define as variáveis independentes (X) removendo colunas irrelevantes para o modelo
x = tabela.drop(columns=["score_credito", "id_cliente"])
# Define a variável dependente (y), que é o valor a ser previsto
y = tabela["score_credito"]

# Importa os modelos de aprendizado de máquina RandomForest e KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier