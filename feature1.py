# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:20:07 2017

@author: www
"""

'''
特征选择
	filter01
	去掉低方差的特征
'''
import pandas as pd
import json

train = pd.read_csv(r"E:\data\alltrain.csv")


y1 = train['label'].copy()
train = train.drop(['id','label'],axis=1)


#评分函数
def score(y_pre, y_true):
    
    P = ((y_pre==0)&(y_true==0)).sum()/((y_pre==0).sum())
    R = ((y_pre==0)&(y_true==0)).sum()/((y_true==0).sum())
    
    return (5*P*R)/(2*P+3*R)*100

#去掉取值变化小的特征  
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
df1 = sel.fit_transform(train)

feature01 = list(train.columns[sel.get_support()])

file = open('feature01.json','w',encoding='utf-8') 
json.dump(feature01,file,ensure_ascii=False)  
file.close() 