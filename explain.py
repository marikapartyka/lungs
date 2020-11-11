#!/usr/bin/env python
# coding: utf-8




from _train import data, X, y, pipeline, pipeline_for_encoded_data,encoded_X
import pandas as pd
import sklearn
import numpy as np
from lime import lime_tabular
import dalex as dx

model = pipeline




X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.80)
model.fit(X_train, y_train)

# create an explainer for the model:
exp = dx.Explainer(model, X, y, label = "Lung's Cancer MLP Pipeline")



# BreakDown and BreakDownInt methods
def BreakDown(number_of_observation):
    bd = exp.predict_parts(pd.DataFrame(X_test.iloc[number_of_observation,:]).T, type='break_down')
    bd.plot()
    
def BreakDownI(number_of_observation):
    bd_interactions = exp.predict_parts(pd.DataFrame(X_test.iloc[number_of_observation,:]).T,
                                        type='break_down_interactions')
    bd_interactions.plot()
    



#SHAP
def Shap(number_of_observation):
    sh = exp.predict_parts(pd.DataFrame(X_test.iloc[number_of_observation,:]).T, type='shap', B = 10)
    sh.plot(bar_width = 16)
Shap(4)

# sh.result.loc[sh.result.B == 0, ]


# #lime
# # preparing categorical_features for lime method
# categorical_features = [3,7,9,10]
# categorical_names = {}

# categorical_names = {}
# for feature in categorical_features:
#     Y = X.copy()
    
#     le = sklearn.preprocessing.LabelEncoder()
#     Y.iloc[:, feature] = Y.iloc[:, feature].astype(str)
#     le.fit(Y.iloc[:, feature])
# #   Y.iloc[:, feature] = le.transform(Y.iloc[:, feature])
#     categorical_names[feature] = le.classes_
   
# stadia = [float('nan'), 'IA1', 'IA2', 'IA3', 'IB', 'IIA', 'IIB', 'IIIA', 'IIIB', 'IVA', 'IVB']
# categorical_names.update({11:np.array(stadia, dtype=object)})
# categorical_features2 = [3,7,9,10,11]
# encoder =  lambda x: model.named_steps["encoder"].transform(x)
# scaler = lambda x: model.named_steps["scaler"].transform(x) 
# predict_fn = lambda x: model.named_steps["nn"].predict_proba(x)
# X_train_enc = encoder(X_train)
# X_test_enc = encoder(X_test)
# X_train_sc = scaler(X_train_enc)
# X_test_sc = scaler(X_test_enc)

# explainer_lime = lime_tabular.LimeTabularExplainer(X_train_sc,class_names=["NO", "YES"],
#                                                    feature_names=X_train.columns,
#                                                   categorical_features=categorical_features2,
#                                                     categorical_names=categorical_names,
#                                                    verbose=False)
# def Lime(number_of_observation):
#     exp_lime = explainer_lime.explain_instance(X_test_sc[number_of_observation],predict_fn)
#     exp_lime.show_in_notebook(show_table=True, show_all=False)
    
# Lime(4)

def CeterisParibus(number_of_observation):
    cp = exp.predict_profile(pd.DataFrame(X_test.iloc[number_of_observation,:]).T)
    cp.plot()
    
CeterisParibus(4)






def VariableImp():
    vi = exp.model_parts()
    vi.plot(max_vars=10)
    
VariableImp()




def PartialDp():
    pdp_num = exp.model_profile(type = 'partial')
    pdp_num.result["_label_"] = 'pdp'
    pdp_num.plot()
    
PartialDp()







