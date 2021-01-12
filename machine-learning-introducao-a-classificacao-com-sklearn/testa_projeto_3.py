from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
# import seaborn as sns

uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados = pd.read_csv(uri)
# print(dados.head())

a_renomear = {'expected_hours' : 'horas_esperadas', 'price' : 'preco', 'unfinished' : 'nao_finalizado'}
dados = dados.rename(columns = a_renomear)
# print(dados.head())

troca = {0: 1, 1: 0}
dados['finalizado'] = dados.nao_finalizado.map(troca)

# sns.scatterplot(x="horas_esperadas", y="preco", hue="finalizado", data=dados)

# sns.relplot(x="horas_esperadas", y="preco", hue="finalizado", col="finalizado", data=dados)

x = dados[['horas_esperadas', 'preco']]
y = dados['finalizado']

SEED = 5
np.random.seed(SEED)
raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25, stratify = y)

scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = SVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

# previsoes_de_chute = np.ones(540)
# acuracia = accuracy_score(teste_y, previsoes_de_chute) * 100
# print("A acurácia do chute foi %.2f%%" % acuracia)

