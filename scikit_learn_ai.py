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

# Importa a função para dividir os dados em conjuntos de treino e teste
from sklearn.model_selection import train_test_split

# Divide os dados em conjuntos de treino (70%) e teste (30%)
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# Importa os modelos de aprendizado de máquina RandomForest e KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Inicializa o modelo Random Forest
modelo_arvoredecisao = RandomForestClassifier()
# Inicializa o modelo K-Nearest Neighbors
modelo_knn = KNeighborsClassifier()

# Treina o modelo Random Forest com os dados de treino
modelo_arvoredecisao.fit(x_treino, y_treino)
# Treina o modelo KNN com os dados de treino
modelo_knn.fit(x_treino, y_treino)

# Realiza previsões no conjunto de teste usando o modelo Random Forest
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
# Realiza previsões no conjunto de teste usando o modelo KNN
previsao_knn = modelo_knn.predict(x_teste)

# Importa a métrica de acurácia para avaliar os modelos
from sklearn.metrics import accuracy_score

# Calcula e exibe a acurácia do modelo Random Forest em porcentagem
accuracy_arvore = accuracy_score(y_teste, previsao_arvoredecisao) * 100
print(f"Acurácia do modelo Random Forest: {accuracy_arvore:.2f}%")

# Calcula e exibe a acurácia do modelo KNN em porcentagem
accuracy_knn = accuracy_score(y_teste, previsao_knn) * 100
print(f"Acurácia do modelo KNN: {accuracy_knn:.2f}%")
