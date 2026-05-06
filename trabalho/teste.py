import pandas as pd
import matplotlib.pyplot as plt
dados=pd.read_csv("class_german_credit.csv")
# print(dados.head())
# print(dados.info())
print(dados.describe())
for column in dados.select_dtypes(include=['number']):
    plt.figure()
    dados[column].plot(kind='hist', bins=20, title='Data Distribution', xlabel=column)
    print(dados[column].value_counts())

for column in dados.select_dtypes(include=['string']):
    plt.figure()
    df=dados[column].groupby(dados[column]).count()
    df.plot.bar()
    plt.style.use('seaborn-v0_8-pastel')
    plt.title('Distr '+column)
    plt.xlabel(column)
    plt.ylabel('Numero de pessoas')
    print(dados[column].value_counts())


plt.show()