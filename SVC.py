# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:17:47 2020

@author: 86136
"""
import pandas as pd
import numpy as np 
from sklearn import svm 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

data_csv = pd.read_csv(r"C:\Users\86136\Desktop\CORINA result.csv")

data = data_csv.drop(['ID'],axis = 1)
data = data.drop(['value'], axis = 1)
#data = data.drop(['SMILES'], axis = 1)

Y = np.array(data_csv['value'])
X = data.values
name_feature = data.columns.values.tolist()
min_max_scaler = preprocessing.MinMaxScaler()

X_process = min_max_scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_process, Y, test_size=0.20, random_state=233)    #训练验证集

svc = svm.SVC(cache_size = 1000,probability = True)

parameters = {'kernel':('linear', 'rbf'),
              'C':[1e-3, 1e-2, 1e-1, 1, 10], 
              'gamma':[0.00001, 0.0001, 0.001, 0.1]}

clf = GridSearchCV(estimator = svc, 
                       param_grid = parameters, scoring='roc_auc',cv=5)

clf.fit(X = X_train, y = Y_train,sample_weight=None) 
print( 'grid search result:',clf.best_params_)
print( 'grid search socore:', clf.best_score_)
y_pred_train = clf.predict(X_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import matthews_corrcoef
import sklearn.metrics
import matplotlib.pyplot as plt
print("Test:\n",sklearn.metrics.classification_report(Y_test, y_pred, target_names = ["0","1"],digits=4))
print("Train:\n",sklearn.metrics.classification_report(Y_train, y_pred_train, target_names = ["0","1"],digits=4))

fpr_test, tpr_test, thresholds = sklearn.metrics.roc_curve(Y_test, clf.predict_proba(X_test)[:,1], pos_label=1)
test_AUC = sklearn.metrics.auc(fpr_test, tpr_test)
fpr_train, tpr_train, thresholds = sklearn.metrics.roc_curve(Y_train, clf.predict_proba(X_train)[:,1], pos_label=1)
train_AUC = sklearn.metrics.auc(fpr_train, tpr_train)
plt.figure(figsize=(12,9))
plt.plot(fpr_test,tpr_test,color='darkorange',label='Test ROC curve (area = %0.4f)' % test_AUC)
plt.plot(fpr_train,tpr_train,color='red',label='Train ROC curve (area = %0.4f)' % train_AUC)
plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig(r"C:\Users\86136\Desktop\result\MACCS_RF_ROC.png")
plt.show()
MCC =  matthews_corrcoef(Y_test, y_pred)
print("test MCC:",MCC)
