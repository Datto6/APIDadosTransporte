import pandas as pd
import numpy as np

from sklearn import tree  # Arvore de decisão e plot tree
from sklearn.metrics import accuracy_score   # Acurácia
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder  # Transformar coluna ordinária
from sklearn.model_selection import train_test_split  # Separar a parte de teste
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Matriz de confusão
import matplotlib.pyplot as plt  # Plot na tabela
from sklearn.impute import KNNImputer

df=pd.read_csv("class_german_credit.csv")

# Sex
df['Sex'] = (df['Sex'] == 'male').astype(int) # female -> 0; male -> 1;

# Housing
encoder = OrdinalEncoder(categories=[['free', 'rent', 'own']])
df['Housing'] = encoder.fit_transform(df[['Housing']])

# Saving Accounts
encoder = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value= -1,categories=[['little', 'moderate', 'quite rich', 'rich']])
imputer = KNNImputer(n_neighbors=5)

df['Saving accounts']=encoder.fit_transform(df[['Saving accounts']]) #converts to numbers

df['Saving accounts'] = df['Saving accounts'].replace(-1, np.nan)

df['Saving accounts']=imputer.fit_transform(df[['Saving accounts']]) #takes care of missing values
df['Saving accounts']+=1 #desloca tudo por um 
df['Saving accounts']=df['Saving accounts'].replace(1,0) #bota 0 de volta pro 0. Agora tabela ta 0, 1.456, 2,3
df['Saving accounts']=df['Saving accounts'].round() #arredonda, porque estamos falando de valores discretos, nao existe decimal
print(df['Saving accounts'].value_counts())

# Checking Account
# encoder = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value= -1,categories=[['little', 'moderate', 'rich']]) 
# imputer = KNNImputer(n_neighbors=5)
# df['Checking account'] = encoder.fit_transform(df[['Checking account']]) #converts to numbers
# df['Checking account'] = imputer.fit_transform(df[['Checking account']]) #takes care of missing values
df=df.drop('Checking account', axis=1)
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


X = df.drop('Risk', axis=1)
y = df['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify = y)

clf = tree.DecisionTreeClassifier(random_state=42)

# Treinamento
clf.fit(X_train, y_train)

# Teste
y_pred = clf.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f'A acurácia do modelo foi de {acuracia*100:.2f}%')