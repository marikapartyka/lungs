# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pickle
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pprint import pprint
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, LabelEncoder, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier

np.random.seed(666)

only_obtain_best_models = True

# %% [markdown]
# # Loading and parsing data

# %%
data = pd.read_csv('./dane_pluca.csv')

# translate to english
data = data.rename(columns={
 'data.urodzenia':                 'date_birth',
 'data.rozpoczecia.leczenia':      'date_start_treatment',
 'data.operacji':                  'date_surgery',
 'zyje':                           'alive',
 'plec':                           'sex',
 'wymiar.guza.x':                  'tumor_size_x',
 'wymiar.guza.y':                  'tumor_size_y',
 'wymiar.guza.z':                  'tumor_size_z',
 'rozpozananie.histopatologiczne': 'histopatological_diagnosis',
 'ile.lat.pali':                   'years_smoking',
 'rodzine.nowotwory.pluc':         'lung_cancer_in_family',
 'objawy.choroby.bol':             'symptoms',
 'stadium.uicc':                   'stadium_uicc',
 'rozpoznanie.wiek':               'age',
 'czas.do.operacji':               'time_to_surgery',
})

data = data.fillna('unknown')

# date variables are transformed to age
def parse_date(date_string):
    #return datetime.date(*map(int, date_string.split('-')))
    return -(datetime.date(*map(int, date_string.split('-'))) - datetime.date(2016, 12, 1)).days//365.25

data.date_birth = data.date_birth.apply(parse_date)
data.date_start_treatment = data.date_start_treatment.apply(parse_date)
data.date_surgery = data.date_surgery.apply(parse_date)

#data['age_start_treatement'] = data.apply(lambda row: (row.date_start_treatment - row.date_birth).days/365.25, axis=1)
#data['age_surgery'] = data.apply(lambda row: (row.date_surgery - row.date_birth).days/365.25, axis=1)

#del data['date_birth']
#del data['date_start_treatment']
#del data['date_surgery']

# YES/NO varaibels are transformed to booleans
for col in ['alive', 'lung_cancer_in_family', 'symptoms']:
  data[col] = data[col].apply(lambda x: x == 'TAK')


# Additional varaibels are defined
data["tumor_volume"] = np.prod(data[["tumor_size_x", "tumor_size_y", "tumor_size_z"]], axis=1)
data["log_tumor_volume"] = np.log(data["tumor_volume"]+1e-4)

del data['histopatological_diagnosis']

X = data.drop(columns='alive')
y = data.alive

X

# %% [markdown]
# # Encoding data

# %%
uicc_stadia = ['unknown', 'IA1', 'IA2', 'IA3', 'IB', 'IIA', 'IIB', 'IIIA', 'IIIB', 'IVA', 'IVB']
uicc_stadium_mapping = dict(zip(uicc_stadia, range(len(uicc_stadia))))

#histopatological_diagnosis_encoder = OneHotEncoder(sparse=False)
#histopatological_diagnosis_encoder.fit(data[['histopatological_diagnosis']])

def encode(X):
  X = X.copy()
  
  X.stadium_uicc = X.stadium_uicc.map(uicc_stadium_mapping)
  X.sex = (X.sex == 'F').astype(int)
  
  #histopatological_diagnosis_encoded = pd.DataFrame(
  #  histopatological_diagnosis_encoder.transform(X[['histopatological_diagnosis']]),
  #  columns = histopatological_diagnosis_encoder.get_feature_names(['HD']),
  #  index=X.index
  #)
  #del X['histopatological_diagnosis']

  #return pd.concat([X, histopatological_diagnosis_encoded], axis=1).astype(float)
  #return X.join(histopatological_diagnosis_encoded)
  return X

X_encoded = encode(X)
X_encoded

# %% [markdown]
# We will examine models with 

# %%
pipelines = dict(
    pipeline_logistic_regression = Pipeline([
      ('encoder', FunctionTransformer(encode)),
      ('scaler', StandardScaler()),
      ('clf', LogisticRegression(max_iter=300))
    ]),

    pipeline_pca_logistic_regression = Pipeline([
      ('encoder', FunctionTransformer(encode)),
      ('scaler', StandardScaler()),
      ('pca', PCA(0.95)),
      ('clf', LogisticRegression(max_iter=300))
    ]),

    pipeline_nn = Pipeline([
      ('encoder', FunctionTransformer(encode)),
      ('scaler', StandardScaler()),
      ('clf', MLPClassifier((10,), learning_rate='adaptive', early_stopping=True))
    ]),

    pipeline_pca_nn = Pipeline([
      ('encoder', FunctionTransformer(encode)),
      ('scaler', StandardScaler()),
      ('pca', PCA(0.95)),
      ('clf', MLPClassifier((10,), learning_rate='adaptive', early_stopping=True))
    ]),

    pipeline_big_nn = Pipeline([
      ('encoder', FunctionTransformer(encode)),
      ('scaler', StandardScaler()),
      ('clf', MLPClassifier((30, 10,), learning_rate='adaptive', early_stopping=True))
    ]),

    pipeline_pca_big_nn = Pipeline([
      ('encoder', FunctionTransformer(encode)),
      ('scaler', StandardScaler()),
      ('pca', PCA(0.95)),
      ('clf', MLPClassifier((30, 10,), learning_rate='adaptive', early_stopping=True))
    ]),

    pipeline_rf = Pipeline([
      ('encoder', FunctionTransformer(encode)),
      ('scaler', StandardScaler()),
      ('clf', RandomForestClassifier())
    ]),

    pipeline_pca_rf = Pipeline([
      ('encoder', FunctionTransformer(encode)),
      ('scaler', StandardScaler()),
      ('pca', PCA(0.95)),
      ('clf', RandomForestClassifier())
    ]),

    pipeline_xgb = Pipeline([
      ('encoder', FunctionTransformer(encode)),
      ('scaler', StandardScaler()),
      ('clf', XGBClassifier())
    ]),

    pipeline_pca_xgb = Pipeline([
      ('encoder', FunctionTransformer(encode)),
      ('scaler', StandardScaler()),
      ('pca', PCA(0.95)),
      ('clf', XGBClassifier())
    ]),
)


# %%
def compare_pipelines(X):
    cv_results = pd.DataFrame()
    for pipeline_name, pipeline in pipelines.items():
        print('\n', pipeline_name)        
        results = cross_validate(pipeline, X, y, scoring=['accuracy', 'roc_auc'], n_jobs=-1)
        cv_results.at[pipeline_name, 'acc'] = np.mean(results['test_accuracy'])
        cv_results.at[pipeline_name, 'auc'] = np.mean(results['test_roc_auc'])
        pprint(results)

    return cv_results


# %%
# full dataset
only_obtain_best_models or compare_pipelines(X)


# %%
# only tumor volume
only_obtain_best_models or compare_pipelines(X.drop(columns=['tumor_size_x', 'tumor_size_y', 'tumor_size_z', 'log_tumor_volume']))


# %%
# only log tumor volume
only_obtain_best_models or compare_pipelines(X.drop(columns=['tumor_size_x', 'tumor_size_y', 'tumor_size_z', 'tumor_volume']))

# %% [markdown]
# LogisticRegression and BigNN yielded best CV scores, regardless of selected input variables.
# 
# They will be examined more throughly on "only log tumor volume" dataset

# %%
pipeline_logistic_regression_cv = Pipeline([
  ('encoder', FunctionTransformer(encode)),
  ('scaler', StandardScaler()),
  ('clf', LogisticRegressionCV(max_iter=1000))
])

pipeline_bigger_nn = Pipeline([
  ('encoder', FunctionTransformer(encode)),
  ('scaler', StandardScaler()),
  ('clf', MLPClassifier((40, 20,), learning_rate='adaptive', early_stopping=True))
])

X_only_log_volume = X.drop(columns=['tumor_size_x', 'tumor_size_y', 'tumor_size_z', 'tumor_volume'])

#if True or not only_obtain_best_models:
if not only_obtain_best_models:
    lrcv_score = cross_validate(
        pipeline_logistic_regression_cv, X_only_log_volume, y, scoring=['accuracy', 'roc_auc'], n_jobs=-1
    )

    biggernn_score = cross_validate(
        pipeline_bigger_nn, X_only_log_volume, y, scoring=['accuracy', 'roc_auc'], n_jobs=-1
    )
    
    print(f"LRCV  mean ACC: {np.mean(lrcv_score['test_accuracy']):.4f}, mean AUC {np.mean(lrcv_score['test_roc_auc']):.4f}")
    print(f"BigNN mean ACC: {np.mean(biggernn_score['test_accuracy']):.4f}, mean AUC {np.mean(biggernn_score['test_roc_auc']):.4f}")

# %% [markdown]
# They are the best models

# %%
X_only_log_volume


# %%
def plot_roc(pipeline):
    X_train, X_test, y_train, y_test = train_test_split(X_only_log_volume, y, test_size=.2, random_state=0)
    pipeline.fit(X_train, y_train)
    y_score = pipeline.predict_proba(X_test)
    fpr, tpr, thr = roc_curve(y_test, y_score[:, 1])
    auc = roc_auc_score(y_test, y_score[:, 1])

    plt.figure(figsize=(9, 6))
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.title(f'Receiver operating curve, AUC = {auc:.2f}')
    
plot_roc(pipeline_logistic_regression_cv)
plot_roc(pipeline_bigger_nn)


# %%
pipeline_logistic_regression_cv_for_encoded_data = deepcopy(pipeline_logistic_regression_cv)
del pipeline_logistic_regression_cv_for_encoded_data.steps[0]
pipeline_logistic_regression_cv_for_encoded_data


# %%
pipeline_bigger_nn_for_encoded_data = deepcopy(pipeline_bigger_nn)
del pipeline_bigger_nn_for_encoded_data.steps[0]
pipeline_bigger_nn_for_encoded_data

