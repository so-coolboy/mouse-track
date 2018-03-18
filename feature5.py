# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:45:36 2017

@author: www
"""



'''
	特征融合
'''
import pandas as pd
import json


train = pd.read_csv(r"E\data\alltrain.csv")
y = train['label'].copy()
train = train.drop(['id','label'],axis=1)

path = ['feature01.json','feature02.json','feature03.json','feature04.json','feature11.json','feature12.json']
fea=[]
for p in path:	
	file = open(p,'r',encoding='utf-8')  
	fea.extend(json.load(file))
	file.close()  
fea_set = set(fea)

feature = {}
for f in fea_set:
	feature[f]=fea.count(f)
	
feature = pd.DataFrame({'count':list(feature.values()), 'fea':list(feature.keys())})
feature=list(feature['fea'][feature['count']>2])

file = open('feature.json','w',encoding='utf-8') 
json.dump(feature,file,ensure_ascii=False)  
file.close() 