# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:30:40 2017

@author: www
"""

'''
特征选择
	filter04
	利用lgbm进行特征选择
'''
import pandas as pd
import json
import lightgbm
import matplotlib.pyplot as plt

train = pd.read_csv(r"E\data\alltrain.csv")
y = train['label'].copy()
train = train.drop(['id','label'],axis=1)

lgbm = lightgbm.LGBMClassifier()
lgbm.fit(train, y)

feat_imp = pd.Series(lgbm.feature_importance(), index=train.columns)
feat_imp = feat_imp[feat_imp>0].sort_values(ascending=False)
feat_imp.plot(kind='bar', title='lgbm   Feature Importances')
plt.ylabel('Feature Importance Score')
plt.figure(figsize=(8,6))

feature4=list(feat_imp[feat_imp>0].index)

file = open('feature04.json','w',encoding='utf-8') 
json.dump(feature4,file,ensure_ascii=False)  
file.close() 