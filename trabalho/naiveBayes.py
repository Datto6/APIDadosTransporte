import pandas as pd
import numpy as np

from sklearn import tree  # Arvore de decisão e plot tree
from sklearn.metrics import accuracy_score   # Acurácia
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, MinMaxScaler,KBinsDiscretizer,FunctionTransformer  # Transformar coluna ordinária
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold  # Separar a parte de teste
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Matriz de confusão
import matplotlib.pyplot as plt  # Plot na tabela
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import ComplementNB
# =========================
# LOAD DATA
# =========================

df = pd.read_csv("class_german_credit.csv")

# =========================
# COLUMN GROUPS
# =========================

binary_cols = ['Sex']
housing_col = ['Housing']
saving_col = ['Saving accounts']
checking_col = ['Checking account']

# =========================
# PREPROCESSING PIPELINE
# =========================
def purposeEncoder(df):
    purpose_encoder = OneHotEncoder(handle_unknown='ignore')

    purpose_encoder.fit(df[['Purpose']]) #fit

    encoded = purpose_encoder.transform(df[['Purpose']]).toarray() # transform

    # turn into DataFrame with proper column names
    encoded_df = pd.DataFrame(encoded, columns=purpose_encoder.get_feature_names_out(['Purpose']))

    df = df.drop(columns=['Purpose']) # drop original column
    df = pd.concat([df, encoded_df], axis=1)
    # Merge the three columns into one
    cols_to_merge = ['Purpose_repairs', 'Purpose_vacation/others', 'Purpose_domestic appliances', 'Purpose_education']
    existing_cols = [c for c in cols_to_merge if c in df.columns] #protects against keyerror, checks if columns exist in df

    if existing_cols:
        df['Purpose_agreggate_others'] = df[existing_cols].max(axis=1) #takes max of row in each existing_col 
        df = df.drop(columns=existing_cols)

    return df

df=purposeEncoder(df)

# target
y = (df['Risk'] == 'good').astype(int)
# print(df.head()) so para olhar valores
# features
X = df.drop(columns='Risk')

preprocessor = ColumnTransformer([
    # Sex -> binary
    ('sex',OrdinalEncoder(categories=[['female', 'male']]), binary_cols),

    # Housing -> ordinal
    ( 'housing',OrdinalEncoder(categories=[['free','rent', 'own']]),housing_col),

    # Saving accounts
    ('saving',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,categories=[['little','moderate','rich','quite rich']]),saving_col),
    # Checking account
    ('checking',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,categories=[['little','moderate','rich']]),checking_col),
    #Coisas de discretizacao, nao funcionou
    # ('age_bins',KBinsDiscretizer(encode='ordinal',quantile_method='linear'), ['Age']),

    # ('duration_bins',KBinsDiscretizer(encode='ordinal',quantile_method='linear'),['Duration']),

    # ('credit_bins',KBinsDiscretizer(encode='ordinal',quantile_method='linear'),['Credit amount'])
], remainder='passthrough')

# =========================
# COMPLETE PIPELINE
# =========================

pipe = Pipeline([
    ('preprocessing', preprocessor), #transforma em numericos p arvore
    ('imputer', KNNImputer(n_neighbors=5)), #imputa missing de saving e checking
    ('rounder',FunctionTransformer(np.round)), #agora arredonda
    ('model', ComplementNB())
])

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
X_train_processed = pipe[:-1].fit_transform(X_train, y_train)
X_test_processed = pipe[:-1].transform(X_test)

feature_names = pipe.named_steps['preprocessing'].get_feature_names_out()

X_train_processed = pd.DataFrame(X_train_processed,columns=feature_names,index=X_train.index)

X_test_processed = pd.DataFrame(X_test_processed,columns=feature_names,index=X_test.index)

print(X_train_processed.head())
print(X_train_processed['saving__Saving accounts'].value_counts())
print(X_train_processed['checking__Checking account'].value_counts()) #isso eh apenas para visualizar o que o kNN fez
print(X_train_processed.info())

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"{acuracia*100:.2f}% accuracy")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()