# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:07:17 2017

@author: www
"""
'''
特征选择
	filter02
	卡方校验 自变量对因变量的相关性
	适合0/1型特征和稀疏矩阵
	要求非负
'''
import pandas as pd
import json
from sklearn.feature_selection import SelectKBest  
'''
For regression: f_regression, mutual_info_regression
For classification: chi2, f_classif, mutual_info_classif
'''
from sklearn.feature_selection import f_classif  

train = pd.read_csv(r"G:\比赛分享\data\alltrain.csv")
test= pd.read_csv(r"G:\比赛分享\data\alltest.csv")

y = train['label']
del train['label']
del train['id']
id_a = test['id']
del test['id']

#评分函数
def score(y_pre, y_true):
    
    P = ((y_pre==0)&(y_true==0)).sum()/((y_pre==0).sum())
    R = ((y_pre==0)&(y_true==0)).sum()/((y_true==0).sum())
    
    return (5*P*R)/(2*P+3*R)*100


file = open('feature01.json','r',encoding='utf-8')  
s = json.load(file) 

train = train[s]
sel = SelectKBest(f_classif, k=100)
data  = sel.fit_transform(train, y)

feature02 = list(train.columns[sel.get_support()])

file = open('feature02.json','w',encoding='utf-8') 
json.dump(feature02,file,ensure_ascii=False)  
file.close()
