# -*- coding: utf-8 -*-
"""
Created on Sat May 20 16:37:44 2017

@author: Yang
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report as cfr
from sklearn.metrics import confusion_matrix as cm
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv(r"G:\比赛分享\data\alltrain.csv")
test= pd.read_csv(r"G:\比赛分享\data\alltest.csv")

id_subm = test['id']
del test['id']
y = train['label']
df = train.drop(['label','id'],axis=1)

sd = StandardScaler()
df = sd.fit_transform(df)
test = sd.transform(test)

def score(y_pre, y_true):
    
    P = ((y_pre==0)&(y_true==0)).sum()/((y_pre==0).sum())
    R = ((y_pre==0)&(y_true==0)).sum()/((y_true==0).sum())
    
    return (5*P*R)/(2*P+3*R)*100

#切分数据集
X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.33, random_state=42)


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train,y_train = smote.fit_sample(X_train,y_train)

pre = pd.DataFrame()

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
print("决策树分数：")
print(dt.score(X_val,y_val))
print("决策树分类报告：")
print(cfr(dt.predict(X_val), y_val), "\n")
print(cm(dt.predict(X_val), y_val), "\n")

pre_dt = dt.predict(test)
pre['dt']=pre_dt


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)
print("随机森林分数：")
print(rf.score(X_val,y_val),"\n")
print(cfr(rf.predict(X_val), y_val), "\n")
print(cm(rf.predict(X_val), y_val), "\n")
pre_rf = rf.predict(test)
pre['rf']=pre_rf

from sklearn.svm import SVC
svm  = SVC()
svm.fit(X_train,y_train)
print("svm分数：")
print(svm.score(X_val,y_val),"\n")
print(cfr(svm.predict(X_val), y_val), "\n")
print(cm(svm.predict(X_val), y_val), "\n")

pre_svm = svm.predict(test)
pre['svm']=pre_svm


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
print("逻辑回归：")
print(lr.score(X_val,y_val),"\n")
print(cfr(lr.predict(X_val), y_val), "\n")
print(cm(lr.predict(X_val), y_val), "\n")

pre_lr = lr.predict(test)
pre['lr']=pre_lr


import xgboost
xgb = xgboost.XGBClassifier()
xgb.fit(X_train,y_train)
print("xgboost分数：")
print(xgb.score(X_val,y_val),"\n")
print(cfr(xgb.predict(X_val), y_val), "\n")
print(cm(xgb.predict(X_val), y_val), "\n")
pre_xgb = xgb.predict(test)
pre['xgb']=pre_xgb

###投票法
p=[]
from scipy.stats import mode
for i in range(len(pre)):
    p.append(mode(pre.ix[i]).mode[0])

#
p = pd.DataFrame({'id':id_subm.values,'restult':p})
subm = p['id'][p['restult']==0]
subm.to_csv('dsjtzs_txfzjh_preliminary_all.txt',index=None)   