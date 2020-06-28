# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:24:15 2019

@author: 86136
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:41:54 2019

@author: lan408-right2
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import  metrics
from sklearn.tree import export_graphviz
import pandas as pd
import numpy as np
data_csv = pd.read_csv(r"C:\Users\86136\Desktop\CYSLTR1_DATA\cysltr1_maccs_1.csv")

data = data_csv.drop(['ID'],axis = 1)
data = data.drop(['new_value'], axis = 1)
data = data.drop(['new_id'], axis = 1)
#data = data.drop(['SMILES'], axis = 1)

Y = np.array(data_csv['new_value'])
X = data.values
name_feature = data.columns.values.tolist()

X_train_1, X_test, Y_train_1, Y_test = train_test_split(X, Y, test_size=0.20, random_state=2333)    #训练验证集

param_test1 = {
'max_features':range(2,105),'n_estimators':range(20,100,5)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier( 
                                  random_state=10, min_samples_split = 2, criterion = 'entropy'), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(X_train_1,Y_train_1)
y_predprob = gsearch1.predict_proba(X_test)[:,1]
print( 'grid search result:',gsearch1.best_params_)
print( 'grid search socore:', gsearch1.best_score_)


rfc = RandomForestClassifier(n_estimators = gsearch1.best_params_["n_estimators"], random_state = 10, criterion = 'entropy')
scores_5=cross_val_score(rfc,X,Y,cv=5,scoring='accuracy')
print('cross5 score:',scores_5.mean())
scores_10 = cross_val_score(rfc,X,Y,cv=10,scoring='accuracy')
print('cross10 score:',scores_10.mean())


#rfc = RandomForestClassifier(n_estimators = 65, max_features = 12 ,random_state = 10, criterion = 'entropy')
rfc.fit(X_train_1,Y_train_1)
y_pred = rfc.predict(X_test)
y_pred_train = rfc.predict(X_train_1)
Y_train = np.array(Y_train_1).squeeze()
Y_test = np.array(Y_test).squeeze()
################################################################################################################################
#验证指标
from sklearn.metrics import matthews_corrcoef
import sklearn.metrics
import matplotlib.pyplot as plt
print("Test:\n",sklearn.metrics.classification_report(Y_test, y_pred, target_names = ["0","1"],digits=4))
print("Train:\n",sklearn.metrics.classification_report(Y_train, y_pred_train, target_names = ["0","1"],digits=4))

fpr_test, tpr_test, thresholds = sklearn.metrics.roc_curve(Y_test, rfc.predict_proba(X_test)[:,1], pos_label=1)
test_AUC = sklearn.metrics.auc(fpr_test, tpr_test)
fpr_train, tpr_train, thresholds = sklearn.metrics.roc_curve(Y_train_1, rfc.predict_proba(X_train_1)[:,1], pos_label=1)
train_AUC = sklearn.metrics.auc(fpr_train, tpr_train)
plt.figure(figsize=(12,9))
plt.plot(fpr_test,tpr_test,color='darkorange',label='Test ROC curve (area = %0.4f)' % test_AUC)
plt.plot(fpr_train,tpr_train,color='red',label='Train ROC curve (area = %0.4f)' % train_AUC)
plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig(r"C:\Users\86136\Desktop\result\CORINA_2D_RF_ROC.png")
plt.show()
MCC_test =  matthews_corrcoef(Y_test, y_pred)
MCC_train =  matthews_corrcoef(Y_train, y_pred_train)

print("test MCC:",MCC_test)
print("train MCC:",MCC_train)
rfc.fit(X,Y)
feature_sorted = sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), name_feature), reverse=True)#round函数指保留几位小数
for i in range(20):
    print(feature_sorted[i])
'''
import os
rfc = RandomForestClassifier(n_estimators =20, random_state = 10, criterion = 'entropy')
rfc.fit(X,Y)

for idx, estimator in enumerate(rfc.estimators_):
    # 导出dot文件
    export_graphviz(estimator,
                    out_file=r"C:\Users\86136\Desktop\result\RF maccs\tree{}.dot".format(idx),
                    feature_names=name_feature,
                    class_names=['high','low'],
                    rounded=True,
                    proportion=False,
                    precision=2,
                    filled=True)
    os.system(r'C:\Users\86136\Desktop\result\RF maccs\dot -Tpng tree{}.dot -o tree{}.png'.format(idx, idx)) #在脚本所在文件夹
import pydot
for i in range(20):
    (graph,) = pydot.graph_from_dot_file(r"C:\Users\86136\Desktop\result\RF maccs\tree{}.dot".format(idx))
    graph.write_png(r"C:\Users\86136\Desktop\result\RF maccs\tree{}.png".format(idx))
'''
########################################################################################################
##maccs部分

data_maccs166 = pd.read_csv(r"C:\Users\86136\Desktop\CYSLTR1_DATA\CYSLT1_maccs.csv")
data = data_maccs166.drop(['ID'],axis = 1)
data = data.drop(['new_value'], axis = 1)
Y = np.array(data_maccs166['new_value'])
X = data.values
name_feature = data.columns.values.tolist()

X_train_1, X_test, Y_train_1, Y_test = train_test_split(X, Y, test_size=0.20, random_state=17)    #训练验证集

param_test1 = {
'max_features':range(10,100),'n_estimators':range(20,150,10)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier( 
                                  random_state=10, min_samples_split = 2, criterion = 'entropy'), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(X_train_1,Y_train_1)
y_predprob = gsearch1.predict_proba(X_test)[:,1]
print( gsearch1.best_params_, gsearch1.best_score_)
time2 = time.time()
#print('cost time:',time2-time1)

rfc = RandomForestClassifier(n_estimators = gsearch1.best_params_["n_estimators"], max_features = gsearch1.best_params_["max_features"], random_state = 10, criterion = 'entropy')
#rfc = RandomForestClassifier(n_estimators = 65, max_features = 12 ,random_state = 10, criterion = 'entropy')
rfc.fit(X_train_1,Y_train_1)
y_pred = rfc.predict(X_test)
y_pred_train = rfc.predict(X_train_1)
AUC_SCORE = metrics.roc_auc_score(Y_test, y_pred)
Y_test = np.array(Y_test).squeeze()
accurancy = accuracy_score(Y_test,y_pred,normalize = True)
#print('auc: ',AUC_SCORE)
print('rfc train score: ',rfc.score(X_train_1, Y_train_1))
print('acc:',accurancy)
feature_sorted = sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), name_feature), reverse=True)#round函数指保留几位小数
uncorrelated = []
#将不相关的添加到uncorrelated内
for m in range(len(feature_sorted)):
    if feature_sorted[m][0] == 0.0:
        uncorrelated.append(feature_sorted[m][1])

print(feature_sorted[0:20])