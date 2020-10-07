from datetime import datetime
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv"
dados = pd.read_csv(uri)

a_renomear = {'mileage_per_year': 'milhas_por_ano', 'model_year': 'ano_do_modelo', 'price': 'preco', 'sold': 'vendido'}
dados = dados.rename(columns=a_renomear)

# troca yes por 1 e no por 0
a_trocar = {'yes': 1, 'no': 0}
dados.vendido = dados.vendido.map(a_trocar)
print(dados.head())

# cria coluna com idade do modelo
ano_atual = datetime.today().year
dados['idade_do_modelo'] = ano_atual - dados.ano_do_modelo

dados['km_por_ano'] = dados.milhas_por_ano * 1.60934

# exclui colunas que não irá usar
dados = dados.drop(columns = ["Unnamed: 0", "milhas_por_ano", "ano_do_modelo"], axis=1)
print(dados.head())

x = dados[["preco", "idade_do_modelo", "km_por_ano"]]
y = dados["vendido"]

SEED = 5
np.random.seed(SEED)
raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25, stratify = y)

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(raw_treino_x), len(raw_teste_x)))

modelo = LinearSVC()
modelo.fit(raw_treino_x, treino_y)
previsoes = modelo.predict(raw_teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

dummy_stratified = DummyClassifier()
dummy_stratified.fit(raw_treino_x, treino_y)
acuracia = dummy_stratified.score(raw_teste_x, teste_y) * 100

# mostra a acurácia conseguida através do dummy
print("A acurácia do dummy foi %.2f%%" % acuracia)

scaler = StandardScaler()
scaler.fit(raw_treino_x)

treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = SVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia com SVC e StandardScaler foi %.2f%%" % acuracia)

