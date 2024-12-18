# Importa a biblioteca pandas para manipulação de dados em formato de tabelas
import pandas as pd

# Carrega o arquivo CSV contendo os dados dos clientes
tabela = pd.read_csv("clientes.csv")
# Exibe informações sobre a estrutura da tabela, como tipos de dados e valores nulos
print(tabela.info())