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
    ('saving',OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=np.nan,
                    categories=[['little','moderate','rich','quite rich']]),
        saving_col
    ),

    # Checking account
    (
        'checking',OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=np.nan,
                    categories=[['little','moderate','rich']]),
        checking_col
    ),

    #Discretization amounts
    ('age_bins',KBinsDiscretizer(encode='ordinal',quantile_method='linear'), ['Age']),

    ('duration_bins',KBinsDiscretizer(encode='ordinal',quantile_method='linear'),['Duration']),

    ('credit_bins',KBinsDiscretizer(encode='ordinal',quantile_method='linear'),['Credit amount'])
], remainder='passthrough')

# =========================
# COMPLETE PIPELINE
# =========================

pipe = Pipeline([
    ('preprocessing', preprocessor), #transforma em numericos p arvore
    ('imputer', KNNImputer(n_neighbors=5)), #imputa missing de saving e checking
    ('rounder',FunctionTransformer(np.round)), #agora arredonda
    ('model', tree.DecisionTreeClassifier(class_weight='balanced',random_state=42))
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

param_grid = {
    'preprocessing__age_bins__n_bins': [2,3,4,5,6],
    'preprocessing__duration_bins__n_bins': [2,3,4,5,6],
    'preprocessing__credit_bins__n_bins': [2,3,4,5,6],

    'preprocessing__age_bins__strategy': ['uniform', 'quantile'],
    'preprocessing__duration_bins__strategy': ['uniform', 'quantile'],
    'preprocessing__credit_bins__strategy': ['uniform', 'quantile'],
    'model__max_depth': [3,5,7,None]
}
cv_strategy = StratifiedKFold(n_splits=5) #classes desbalanceadas tem que manter stratified
grid = GridSearchCV(
    pipe,
    param_grid,
    scoring='accuracy',
    cv=cv_strategy
)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"{acuracia*100:.2f}% best model accuracy")
best_cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=best_cm)
disp.plot(cmap='Blues')
plt.title('Best Confusion Matrix')
plt.show()


