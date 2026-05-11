import pandas as pd
import numpy as np

from sklearn import tree  # Arvore de decisão e plot tree
from sklearn.metrics import accuracy_score   # Acurácia
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, MinMaxScaler,KBinsDiscretizer,FunctionTransformer  # Transformar coluna ordinária
from sklearn.model_selection import train_test_split,GridSearchCV  # Separar a parte de teste
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Matriz de confusão
import matplotlib.pyplot as plt  # Plot na tabela
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df=pd.read_csv("class_german_credit.csv")

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
    # concatenate
    df = pd.concat([df, encoded_df], axis=1)
    return df

df=purposeEncoder(df)

# target
y = (df['Risk'] == 'good').astype(int)
print(df.head())
# features
X = df.drop(columns='Risk')

preprocessor = ColumnTransformer([

    # Sex -> binary
    ('sex',OrdinalEncoder(categories=[['female', 'male']]), binary_cols),

    # Housing -> ordinal
    ( 'housing',
        OrdinalEncoder(categories=[['free', 'rent', 'own']]),
        housing_col
    ),


    # Saving accounts
    (
        'saving',
        Pipeline([
            (
                'encoder',
                OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=np.nan,
                    categories=[[
                        'little',
                        'moderate',
                        'rich',
                        'quite rich'
                    ]]
                )
            ),
            (
                'imputer',
                KNNImputer(n_neighbors=5)
            )
        ]),
        saving_col
    ),

    # Checking account
    (
        'checking',
        Pipeline([
            (
                'encoder',
                OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=np.nan,
                    categories=[[
                        'little',
                        'moderate',
                        'rich'
                    ]]
                )
            ),
            (
                'imputer',
                KNNImputer(n_neighbors=5)
            )
        ]),
        checking_col
    )

], remainder='passthrough')

# =========================
# COMPLETE PIPELINE
# =========================

pipe = Pipeline([
    ('preprocessing', preprocessor),
    # ('imputer', KNNImputer(n_neighbors=5)), implementar isso aqui na proxima
    # ('model', DecisionTreeClassifier(...))
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
X_train_processed =  pd.DataFrame(pipe.fit_transform(X_train), columns=pipe.get_feature_names_out(), index=X_train.index) #retorna dataframe com Savings preenchidos com kNN
X_test_processed = pd.DataFrame(pipe.transform(X_test), columns=pipe.get_feature_names_out(), index=X_test.index)
print(X_train_processed['checking__Checking account'].value_counts())
clf = tree.DecisionTreeClassifier(class_weight='balanced',random_state=42)
            # Treinamento
clf.fit(X_train_processed, y_train)
            # Teste
y_pred = clf.predict(X_test_processed)
acuracia = accuracy_score(y_test, y_pred)
print(f"{acuracia*100:.2f}%")
best_cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=best_cm)
disp.plot(cmap='Blues')
plt.title('Best Confusion Matrix')
print(X_train_processed.head())
plt.show()
# =========================
# FIT + TRANSFORM
# =========================




