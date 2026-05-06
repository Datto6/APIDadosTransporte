import pandas as pd
import numpy as np

from sklearn import tree  # Arvore de decisão e plot tree
from sklearn.metrics import accuracy_score   # Acurácia
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, MinMaxScaler,KBinsDiscretizer  # Transformar coluna ordinária
from sklearn.model_selection import train_test_split  # Separar a parte de teste
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Matriz de confusão
import matplotlib.pyplot as plt  # Plot na tabela
from sklearn.impute import KNNImputer

df=pd.read_csv("class_german_credit.csv")


# Sex
df['Sex'] = (df['Sex'] == 'male').astype(int) # female -> 0; male -> 1;
# df=df.drop('Sex', axis=1)

# Housing
encoder = OrdinalEncoder(categories=[['free', 'rent', 'own']])
df['Housing'] = encoder.fit_transform(df[['Housing']])

# Purpose
purpose_encoder = OneHotEncoder(handle_unknown='ignore')

purpose_encoder.fit(df[['Purpose']]) #fit

encoded = purpose_encoder.transform(df[['Purpose']]).toarray() # transform

# turn into DataFrame with proper column names
encoded_df = pd.DataFrame(encoded, columns=purpose_encoder.get_feature_names_out(['Purpose']))

df = df.drop(columns=['Purpose']) # drop original column

# concatenate
df = pd.concat([df, encoded_df], axis=1)

# Risk
df['Risk'] = (df['Risk'] == 'good').astype(int) # bad -> 0; good -> 1;



#Encodando Checking e Savings
encoder = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value= np.nan,categories=[['little', 'moderate','rich','quite rich']]) #encoder de savings


df['Saving accounts']=encoder.fit_transform(df[['Saving accounts']]) #converts to numbers

encoder = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value= np.nan,categories=[['little', 'moderate', 'rich']]) #encoder de checking
df['Checking account'] = encoder.fit_transform(df[['Checking account']]) #converts to numbers

# Separando dados de treino e de teste
X = df.drop('Risk', axis=1)
y = df['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify = y)

df_train=pd.concat([X_train,y_train], axis=1) #juntando de novo df de treino e de teste, para preencher dados
df_test=pd.concat([X_test,y_test], axis=1)

# Preenchendo Saving accounts
print(df_train['Saving accounts'].value_counts())

features = df.drop(columns=['Checking account']).columns
imputer = KNNImputer(n_neighbors=5)
imputed = imputer.fit_transform(df_train.drop(columns=['Checking account'])) #exclui o checking account do calculo do kNN
imputed_df = pd.DataFrame(imputed, columns=features, index=df_train.index) #retorna dataframe com Savings preenchidos com kNN

df_train['Saving accounts'] = imputed_df['Saving accounts']
df_train['Saving accounts']=df_train['Saving accounts'].round() #arredonda, porque estamos falando de valores discretos, nao existe decimal
X_train['Saving accounts']=df_train['Saving accounts']

imputed=imputer.transform(df_test.drop(columns='Checking account'))
imputed_df = pd.DataFrame(imputed, columns=features, index=df_test.index) #retorna dataframe com Savings preenchidos com kNN

df_test['Saving accounts'] = imputed_df['Saving accounts']
df_test['Saving accounts']=df_test['Saving accounts'].round() #arredonda, porque estamos falando de valores discretos, nao existe decimal

X_test['Saving accounts']=df_test['Saving accounts']

print(df_train['Saving accounts'].value_counts())


# Checking Account--> preenchendo
print(df_train['Checking account'].value_counts())

imputer = KNNImputer(n_neighbors=5) #inicializando kNN 
features = df.columns

imputed = imputer.fit_transform(df_train) #usa dataframe todo inclusive o savings para preencher com kNN
imputed_df = pd.DataFrame(imputed, columns=features, index=df_train.index) #transforma retorno de kNN em dataframe

df_train['Checking account'] = imputed_df['Checking account'] #coloca valores la dentro
df_train['Checking account']=df_train['Checking account'].round() #arredonda, porque estamos falando de valores discretos, nao existe decimal
X_train['Checking account']=df_train['Checking account']

imputed=imputer.transform(df_test)
imputed_df = pd.DataFrame(imputed, columns=features, index=df_test.index) #retorna dataframe com Savings preenchidos com kNN

df_test['Checking account'] = imputed_df['Checking account']
df_test['Checking account']=df_test['Checking account'].round() #arredonda, porque estamos falando de valores discretos, nao existe decimal

X_test['Checking account']=df_test['Checking account']


print(df_train['Checking account'].value_counts())

max=-1
best_cm = None
best_params = None
best_model = None
for i in range(2,40):
    age_discretizer = KBinsDiscretizer(
        n_bins=i,
        encode='ordinal',
        strategy='uniform'
    )
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    # usar copias

    X_train_copy['Age'] = age_discretizer.fit_transform(X_train[['Age']])
    X_test_copy['Age'] = age_discretizer.transform(X_test[['Age']])
    for j in range(2,40):
        credit_discretizer = KBinsDiscretizer(n_bins=j,encode='ordinal',strategy='uniform')
        X_train_copy['Credit amount'] = credit_discretizer.fit_transform(X_train[['Credit amount']])
        X_test_copy['Credit amount'] = credit_discretizer.transform(X_test[['Credit amount']])
        clf = tree.DecisionTreeClassifier(class_weight='balanced',random_state=42)

        # Treinamento
        clf.fit(X_train_copy, y_train)

        # Teste
        y_pred = clf.predict(X_test_copy)

        acuracia = accuracy_score(y_test, y_pred)
        if acuracia>max:
            max=acuracia
            print(f'A acurácia do modelo foi de {acuracia*100:.2f}% com {i} bins de idade e {j} bins de credit amount')
            max = acuracia
            best_cm = confusion_matrix(y_test, y_pred)
            best_params = (i, j)
            best_model = clf

print(f"Best accuracy: {max*100:.2f}% with Age={best_params[0]} and Credit={best_params[1]}")
plt.figure(figsize=(4,4))
disp = ConfusionMatrixDisplay(confusion_matrix=best_cm)
disp.plot(cmap='Blues')
plt.title('Best Confusion Matrix')


importancia = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Importância das colunas:\n", importancia)

plt.show()