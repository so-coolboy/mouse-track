# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:22:35 2017

@author: Yang
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import  SVC
from sklearn.preprocessing import StandardScaler
import lightgbm 
from sklearn.ensemble import GradientBoostingClassifier
class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
    def fit_predict(self, X, y, T):#输入训练集  标签  测试集
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))
								
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
								
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf[1].fit(X_train, y_train)
                y_pred = clf[1].predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf[1].predict(T)[:]

            S_test[:, i] = S_test_i.mean(1)
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred
								
								
train = pd.read_csv(r"G:\比赛分享\data\alltrain.csv")
test= pd.read_csv(r"G:\比赛分享\data\alltest.csv")

id_subm = test['id']
del test['id']
y = train['label']
train = train.drop(['label','id'],axis=1)

sd = StandardScaler()
train = sd.fit_transform(train)
test = sd.transform(test)


clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = lightgbm.LGBMClassifier()
clf4 = SVC()
clf5 = GradientBoostingClassifier()

basemodel = [
            ['lr', clf1],
            ['rf', clf2],
            ['lgb', clf3],
            ['svc', clf4],
            ['gdbt',clf5],
  
        ]
stacker = xgb.XGBClassifier()


en = Ensemble(5, stacker, basemodel)
pre = en.fit_predict(train,y,test)



