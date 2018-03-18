# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 10:45:13 2017

@author: www
"""

import pandas as pd
import numpy as np

import os
import warnings
import json
warnings.filterwarnings("ignore")

#数据清洗，合并训练集和测试集
def get_data():
	#set path
	path = r'.\data'
	train_path = os.path.join(path, 'dsjtzs_txfz_training.txt')
	test_path = os.path.join(path, 'dsjtzs_txfz_test1.txt')
	testB_path = os.path.join(path, 'dsjtzs_txfz_testB.txt')
	#load data
	train = pd.read_csv(train_path, sep=' ', names=['id','point', 'target','label']).ix[:100]#实际运行 去掉 .ix[:100]																					
	test = pd.read_csv(test_path, sep=' ', names=['id','point', 'target']).ix[:100]
	testB =pd.read_csv(testB_path, sep=' ', names=['id','point', 'target']).ix[:100]
	    
	#合并数据集
	label=train['label'].copy()
	train.drop('label',axis=1,inplace=True)
	df = pd.concat([train, test, testB], ignore_index=True)
	id_data = df['id'].copy()
	df.drop('id',axis=1,inplace=True)
	
	train_len = len(train)
	test_len = len(test)
	
	global path
	global id_data
	global label
	global train_len
	global test_len
	return df

#原始数据处理
def data_process(data):
	data['point'] = data['point'].apply(lambda x:[list(map(float,point.split(','))) for point in x.split(';')[:-1]])
	data['target'] = data['target'].apply(lambda x: list(map(float,x.split(","))))	
	#提取 x坐标 y坐标 t 目标点x坐标  目标点y坐标 
	df = pd.DataFrame()
	df['x'] = data['point'].apply(lambda x:np.array(x)[:,0])
	df['y'] = data['point'].apply(lambda x:np.array(x)[:,1])
	df['t'] = data['point'].apply(lambda x:np.array(x)[:,2])
	df['target_x'] = np.array(data['target'].tolist())[:,0]
	df['target_y'] = np.array(data['target'].tolist())[:,1]

	return df	
#差分处理
def data_diff(data,name_list):
	for name in name_list:
		data['diff_'+name] = data[name].apply(lambda x: pd.Series(x).diff().dropna().tolist())
		data['diff_'+name] = data['diff_'+name].apply(lambda x: [0] if x==[] else x)#!!注意 一个点的情况
	return data	
	
#获取距离数据
def get_dist(data):
	dist_target = []
	dist = []
	dist_x_target = []
	dist_y_target = []

	#各点与目标点的距离
	for x,y,target_x,target_y in zip(data['x'],data['y'],data['target_x'],data['target_y']):
		dist_target.append(np.sqrt((x-target_x)**2 + (y-target_y)**2))	
	#两点之间的距离
	for x,y in zip(data['diff_x'], data['diff_y']):
		dist.append(np.sqrt(np.array(x)**2+np.array(y)**2))	
	#各点x坐标与目标点x坐标的距离
	for x,target_x in zip(data['x'], data['target_x']):
		dist_x_target.append(np.sqrt((x-target_x)**2))
	#各点y坐标与目标点y坐标的距离
	for y,target_y in zip(data['y'], data['target_y']):
		dist_y_target.append(np.sqrt((y-target_y)**2))
		
	data['dist_target'] = dist_target
	data['dist'] = dist
	data['dist_x_target'] = dist_x_target
	data['dist_y_target'] = dist_y_target

	return data
	
#获取速度数据
def get_v(data):
	v = []
	v_x = []
	v_y = []
	#获取两点之间的速度
	for dist, t in zip(data['dist'], data['diff_t']):
		v0 = dist/t
		v0 = list(map(lambda x: 0 if x==np.inf or x==-np.inf else x, v0))#!! 注意除数为0的情况
		v.append(v0)
	#获取两点x坐标之间的速度
	for x, t in zip(data['diff_x'], data['diff_t']):
		v1 = np.array(x)/np.array(t)
		v1 = list(map(lambda x: 0 if x==np.inf or x==-np.inf or  np.isnan(x) else x, v1))
		v_x.append(v1)
	#获取两点之间的速度
	for y, t in zip(data['diff_y'], data['diff_t']):
		v2 = np.array(y)/np.array(t)
		v2 = list(map(lambda x: 0 if x==np.inf or x==-np.inf or np.isnan(x) else x, v2))
		v_y.append(v2)
		
	data['v'] = v
	data['v_x'] = v_x
	data['v_y'] = v_y
	
	return data
	
#获取加速度数据
def get_a(data):
	a = []
	a_x = []
	a_y = []
	#获取两点之间的加速度
	for v, t in zip(data['diff_v'], data['diff_t']):
		v = np.array(v)
		t = np.array(t)
		a_t = (t[:-1] + t[1:])/2
		a0 = v/a_t	
		a0 =	list(map(lambda x: 0 if x==np.inf or x==-np.inf else x, a0))#!! 注意除数为0的情况
		#!!注意 列表为空
		if a0==[] : 	
			a0=[0]
		a.append(a0)	
	#获取两点x坐标之间的加速度
	for v_x, t in zip(data['diff_v_x'], data['diff_t']):
		v_x = np.array(v_x)
		t = np.array(t)
		a_t = (t[:-1] + t[1:])/2
		a1 = v_x/a_t	
		a1 =	list(map(lambda x: 0 if x==np.inf or x==-np.inf else x, a1))#!! 注意除数为0的情况
		if a1==[] : 	
			a1=[0]
		a_x.append(a1)					
	#获取两点x坐标之间的加速度
	for v_y, t in zip(data['diff_v_y'], data['diff_t']):
		v_y = np.array(v_y)
		t = np.array(t)
		a_t = (t[:-1] + t[1:])/2
		a2 = v_y/a_t	
		a2 =	list(map(lambda x: 0 if x==np.inf or x==-np.inf else x, a2))#!! 注意除数为0的情况
		if a2==[] : 	
			a2=[0]
		a_y.append(a2)					
	
	data['a'] = a
	data['a_x'] = a_x
	data['a_y'] = a_y

	return data	
def get_feature(data, name):
	dfGroup=pd.DataFrame()
	dfGroup[name+'_start'] = data.apply(lambda x: x[0])
	dfGroup[name+'_end'] = data.apply(lambda x: x[len(x)-1])
	dfGroup[name+'_max'] = data.apply(lambda  x: max(x))
	dfGroup[name+'_min'] = data.apply(lambda  x: min(x))
	dfGroup[name+'_ptp'] = dfGroup[name+'_max'].sub(dfGroup[name+'_min'])
	dfGroup[name+'_mean'] = data.apply(lambda  x: np.mean(x))
	dfGroup[name+'_std'] = data.apply(lambda  x: np.std(x))
	dfGroup[name+'_cv'] = dfGroup[name+'_std'].div(dfGroup[name+'_mean'], fill_value=0)
	dfGroup[name+'_cv'] = dfGroup[name+'_cv'].replace([np.inf,-np.inf],[0,0])
	dfGroup[name+'_cv'] = dfGroup[name+'_cv'].fillna(0)
	dfGroup[name+'_Q1'] = data.apply(lambda  x: np.percentile(x, 0.25))
	dfGroup[name+'_Q2'] = data.apply(lambda  x: np.percentile(x, 0.5))
	dfGroup[name+'_Q3'] = data.apply(lambda  x: np.percentile(x, 0.75))
	dfGroup[name+'_interRan'] = dfGroup[name+'_Q3'].sub(dfGroup[name+'_Q1'])
	dfGroup[name+'_skew'] = data.apply(lambda  x: pd.Series(x).skew()).fillna(0)
	dfGroup[name+'_kurt'] = data.apply(lambda  x: pd.Series(x).kurt()).fillna(0)
    
	return dfGroup

def get_point_feature(df):
    
	point_x = get_feature(df['x'], 'x')
	point_y = get_feature(df['y'], 'y')
	point = pd.concat([point_x, point_y], axis=1)
    
	point['target_x'] = df['target_x'].values
	point['target_y'] = df['target_y'].values
    

	return point
    
def get_dist_feature(df):
	dist_target = get_feature(df['dist_target'], 'dist_target')
	dist_x_target =  get_feature(df['dist_x_target'], 'dist_x_target')
	dist_y_target =  get_feature(df['dist_y_target'], 'dist_y_target')
	diff =  get_feature(df['dist'], 'dist')
	diff_x =  get_feature(df['diff_x'], 'diff_x')
	diff_y =  get_feature(df['diff_y'], 'diff_y')
    
	dist = pd.concat([dist_target, dist_x_target, dist_y_target,
                      diff, diff_x, diff_y], axis=1)

	return dist

def get_time_feature(df):
	t = get_feature(df['t'], 't')
	t_diff = get_feature(df['diff_t'], 'diff_t')
    
	t = pd.concat([t, t_diff], axis=1)

	return t

def get_v_feature(df):
	v_x = get_feature(df['v_x'], 'v_x')
	v_y = get_feature(df['v_y'], 'v_y')
	v = get_feature(df['v'], 'v')
	v_diff_x = get_feature(df['diff_v_x'], 'diff_v_x')
	v_diff_y = get_feature(df['diff_v_y'], 'diff_v_y')
	v_diff = get_feature(df['diff_v'], 'diff_v')
    
	v = pd.concat([v_x, v_y, v,
                   v_diff_x, v_diff_y, v_diff], axis=1)

	return v
    
def get_a_feature(df):
	a_x = get_feature(df['a_x'], 'a_x')
	a_y = get_feature(df['a_y'], 'a_y')
	a = get_feature(df['a'], 'a')
    
	a = pd.concat([a_x, a_y, a], axis=1)
	
	with open('a_feature.json', 'w',encoding='utf-8')as f:
		json.dump(list(a.columns), f, ensure_ascii=False)
	file = open('a_feature.json','w',encoding='utf-8') 
	json.dump(list(a.columns),file,ensure_ascii=False)  
	file.close() 

	return a
def get_other_feature(data):
	dfGroup=pd.DataFrame()
	dfGroup['point_count'] = data['x'].apply(lambda x: len(x))
	dfGroup['x_back_num'] = data['diff_x'].apply(lambda x: min( (np.array(x) > 0).sum(), (np.array(x) < 0).sum()))
	dfGroup['y_back_num'] = data['diff_y'].apply(lambda x: min( (np.array(x) > 0).sum(), (np.array(x) < 0).sum()))
	dfGroup['x_equal_0'] = data['diff_x'].apply(lambda x:  (np.array(x) == 0).sum())
	dfGroup['y_equal_0'] = data['diff_y'].apply(lambda x:  (np.array(x) == 0).sum())
	dfGroup['equal_0'] = data['dist'].apply(lambda x: (np.array(x) == 0).sum())
	return dfGroup
	
def make_data(df):
	df = data_process(df)
	df = data_diff(df, ['x', 'y', 't'])
	df = get_dist(df)
	df = get_v(df)
	df = data_diff(df, ['v', 'v_x', 'v_y'])
	df = get_a(df)
	
    
	point = get_point_feature(df[['x','y','target_x','target_y']])
	dist = get_dist_feature(df[['diff_x', 'diff_y','dist_target', 'dist', 'dist_x_target', 'dist_y_target']])
	t = get_time_feature(df[['t','diff_t']])
	v = get_v_feature(df[['v', 'v_x','v_y', 'diff_v', 'diff_v_x','diff_v_y']])
	a = get_a_feature(df[['a','a_x', 'a_y']])
	other = get_other_feature(df)
	
	df1 = pd.concat([point, dist, t, v,a,other], axis=1)
	return df1.fillna(0)    

def save_df(df,name):
	global path
	global id_data
	global label
	global train_len
	global test_len
	df['id'] = id_data
	train = df.ix[:train_len-1,:]
	train['label'] = label
	test = df.ix[train_len:train_len+test_len-1,:]
	testB = df.ix[train_len+test_len:,:]

	train.to_csv(path+"\\" +name+ "train.csv", index=None)
	test.to_csv(path+"\\" +name+"test.csv", index=None)
	testB.to_csv(path+"\\" +name+"testB.csv", index=None)
	
		
if __name__ == '__main__':
	
	df = get_data()
	df = make_data(df)

	save_df(df,'all')
