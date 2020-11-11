# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import pickle
import datetime
from pprint import pprint

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, LabelEncoder
from sklearn.neural_network import MLPClassifier

np.random.seed(666)


# %%
# read data
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
data["volume"] = np.prod(data[["tumor_size_x", "tumor_size_y", "tumor_size_z"]], axis=1)
print(data.columns)
data.head()


# %%
# features transformation

date_features = ['date_birth', 'date_start_treatment', 'date_surgery']
categorical_features = ['sex', 'histopatological_diagnosis', 'lung_cancer_in_family', 'symptoms']

# numeric features are left unchanged

# all dates are stored as number of days from 1920-01-01 as earliest date in the dataset is 1921-01-21
start_date = datetime.date(1920, 1, 1)
def days_from_start_date(date_string):
  date = datetime.date(*map(int, date_string.split('-')))
  return (date-start_date).days

# stadium is coded by increasing integers 0 being N/A
# severity increases with number
stadia = [float('nan'), 'unknown', 'IA1', 'IA2', 'IA3', 'IB', 'IIA', 'IIB', 'IIIA', 'IIIB', 'IVA', 'IVB']
stadium_mapping = dict(zip(stadia, range(len(stadia))))

# all other categorical variables are encoded as integers
# order is irrelevant
categorical_encoders = dict()
for feature in categorical_features:
  encoder = LabelEncoder()
  encoder.fit(data[feature].astype(str))
  categorical_encoders[feature] = encoder

def encode(X):
  X = X.copy()
  
  X.stadium_uicc = X.stadium_uicc.map(stadium_mapping)
  for feature in X:
    if feature in date_features:
      X[feature] = X[feature].map(days_from_start_date)
    if feature in categorical_features:
      X[feature] = categorical_encoders[feature].transform(X[feature].astype(str))
  return X

encoded_data = encode(data)
encoded_data


# %%
X = data.drop(columns='alive')
y = data.alive == 'TAK'

pipeline = Pipeline([
  ('encoder', FunctionTransformer(encode)),
  ('scaler', StandardScaler()),
  ('nn', MLPClassifier((10,), verbose=False, early_stopping=True))
])

pipeline.fit(X, y)

encoder = pipeline['encoder']
encoded_X = encoder.transform(X)
pipeline_for_encoded_data = Pipeline([
  ('scaler', StandardScaler()),
  ('nn', MLPClassifier((10,), verbose=False, early_stopping=True))
])
pipeline_for_encoded_data.fit(encoded_X, y)

# features without correlation + volume
X_w_corr = encoded_X.drop(columns=["date_birth", "tumor_size_x", "tumor_size_y", "date_surgery", "volume"])
model_w_corr = pipeline_for_encoded_data.fit(X_w_corr, y)

#features without tumor size + volume

X_v = encoded_X.drop(columns = ["date_birth", "tumor_size_x", "tumor_size_y", "tumor_size_z","date_surgery"])
model_v = pipeline_for_encoded_data.fit(X_v, y)

# %%
pickle.dump(pipeline, open('model.pickle', 'wb'))


# %%
from sklearn.model_selection import cross_validate, KFold
# CV acc
accs = []
for train_index, test_index in KFold(shuffle=True).split(X):
    pipeline.fit(X.loc[train_index], y.loc[train_index])
    acc = (pipeline.predict(X.loc[test_index]) == y.loc[test_index]).mean()
    accs.append(acc)

print(np.mean(accs), '+-', np.std(accs))


# %%
# ROC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
#pipeline.fit(X_train, y_train)
y_score = pipeline.predict_proba(X_test)
fpr, tpr, thr = roc_curve(y_test, y_score[:, 1])
auc = roc_auc_score(y_test, y_score[:, 1])
print(auc)


# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 6))
plt.plot(fpr, tpr)
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title(f'Receiver operating curve, AUC = {auc:.2f}')
plt.savefig('auc.png')


# %%
if __name__ == "__main__":
    
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt


    def do_the_thing(classifier, name):
        # prepare data
        X = data.drop(columns='alive')
        y = data.alive == 'TAK'
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

        # define pipeline
        pipeline = Pipeline([
            ('encoder', FunctionTransformer(encode)),
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])

        # compute CV acc
        accs = []
        for train_index, test_index in KFold(shuffle=True).split(X):
            pipeline.fit(X.loc[train_index], y.loc[train_index])
            acc = (pipeline.predict(X.loc[test_index]) == y.loc[test_index]).mean()
            accs.append(acc)

        cv_acc = np.mean(accs)
        cv_acc_std = np.std(accs)
        print(name)
        print(cv_acc, '+-', cv_acc_std)

        # compute AUC on validation set (20% of whole)
        pipeline.fit(X_train, y_train)
        y_score = pipeline.predict_proba(X_test)
        fpr, tpr, thr = roc_curve(y_test, y_score[:, 1])
        auc = roc_auc_score(y_test, y_score[:, 1])
        print(auc)

        # plot ROC
        plt.figure(figsize=(9, 6))
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive rate')
        plt.ylabel('True Positive rate')
        plt.title(f'ROC for {name}, AUC = {auc:.2f}, CV ACC = {cv_acc:.2f}+-{cv_acc_std:.2f}')
        plt.savefig(f'auc_{name}.png')

    do_the_thing(
        MLPClassifier(verbose=False, early_stopping=True), 'nn'
    )


    # %%
    do_the_thing(
        MLPClassifier((10,), verbose=False), 'nn_10'
    )

    do_the_thing(
        MLPClassifier((30,), verbose=False), 'nn_30'
    )

    do_the_thing(
        MLPClassifier((10, 10), verbose=False), 'nn_10_10'
    )


    # %%
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    do_the_thing(
        LogisticRegression(), 'Logistic Regression'
    )

    do_the_thing(
        LogisticRegressionCV(), 'Logistic Regression CV'
    )

    do_the_thing(
        RandomForestClassifier(), 'RF'
    )

    do_the_thing(
        XGBClassifier(), 'xgb'
    )


