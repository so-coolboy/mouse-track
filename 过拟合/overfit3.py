# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 09:20:10 2017

@author: Yang
"""

'''
判断所需特征的个数
	特征的数量与模型误差关系
	大概50个左右的特征
	特征的随机选取造成误差波动很大
'''
import pandas as pd
import numpy as np
import random
import xgboost
import matplotlib.pylab as plt

train = pd.read_csv(r"G:\比赛分享\data\alltrain.csv")
test= pd.read_csv(r"G:\比赛分享\data\alltest.csv")
y_s = pd.read_csv(r'G:\比赛分享\过拟合\y_s.csv')
y_subm = y_s['label']

y = train['label']
del train['label']
del train['id']
del test['id']

feature = list(train.columns)

def score(y_pre, y_true):
    
    P = sum((y_pre==0)&(y_true==0)) / (sum(y_pre==0))
    R = sum((y_pre==0)&(y_true==0)) / (sum(y_true==0))
    
    return (5*P*R)/(2*P+3*R)*100
xgb_params = {"objective": "reg:linear", "eta": 0.01, "max_depth": 8, "seed": 42, "silent": 1}
num_rounds = 1000

score_train=[]
score_test=[]
for i in np.arange(1, len(feature)-1):
	print("train..",i)
	fea = random.sample(feature,i)
	train1 = train[fea]
	test1 = test[fea]	
	xgb = xgboost.XGBClassifier()
	xgb.fit(train1,y)
	y_train = xgb.predict(train1)
	y_test = xgb.predict(test1)
	
	score_train.append(100-score(y_train, y))
	score_test.append(100-score(y_test, y_subm))
	
	
plt.xlabel("feature_num")
plt.ylabel("Score")
plt.ylim(0.0, 100)
plt.xlim(0,300)
plt.figure(figsize=(4,16))
plt.plot(score_train,label="Training score", color="r")
plt.plot(score_test,label="Test score", color="g")
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.save("2.png")
plt.show()
