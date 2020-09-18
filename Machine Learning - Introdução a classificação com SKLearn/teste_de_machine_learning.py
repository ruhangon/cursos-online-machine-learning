from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# cachorro ou porco
# features (1 sim, 0 não)
# pelo longo? 
# perna curta?
# faz auau?

porco1=[0, 1, 0]
porco2=[0, 1, 1]
porco3=[1, 1, 0]

cachorro1=[0, 1, 1]
cachorro2=[1, 0, 1]
cachorro3=[1, 1, 1]

# 1 = porco
# 0 = cachorro
treino_x=[porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y=[1, 1, 1, 0, 0, 0]

model=LinearSVC()
model.fit(treino_x, treino_y)
# procura aprender a classificar com base nos dados passados
# como é um aprendizado supervisionado sendo estudado também é necessário passar as classes, que contém as informações de quais são quais

misterioso1=[1, 1, 1]
misterioso2=[1, 1, 0]
misterioso3=[0, 1, 1]
teste_x=[misterioso1, misterioso2, misterioso3]
teste_y=[0, 1, 1]

previsoes = model.predict(teste_x)

print(previsoes)
print(teste_y)
print(previsoes==teste_y)

taxa_de_acerto = accuracy_score(teste_y, previsoes)
print("taxa de acerto: %.2f " % (taxa_de_acerto * 100))

