from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
dados = pd.read_csv(uri)
print(dados.head())

x = dados[["home","how_it_works","contact"]]
y = dados[["bought"]]

# treino_x = x[:75]
# usará cerca de 75% do banco de dados para treinar o algoritmo
# treino_y = y[:75]

# teste_x=x[75:]
# teste_y=y[75:]

SEED = 20
# faz com que a aleatoriedade escolhida não seja mais tão aleatória
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state = SEED, test_size = 0.25, stratify = y)
# stratify irá estratificar os dados proporcionalmente de acordo com y

print("treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

model = LinearSVC()
model.fit(treino_x, treino_y)

previsoes = model.predict(teste_x)

acerto = accuracy_score(teste_y, previsoes) * 100

print("o percentual de acerto foi de %.2f%%" % acerto)

# print(treino_y.value_counts())
# print(teste_y.value_counts())
# verifica se dados de treino e teste sobre quantos compraram, nesse caso, estão proporcionais. Descobrimos que a princípio não estão
# stratify consegue resolver esse problema da proporcionalidade

