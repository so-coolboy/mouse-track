# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:07:33 2017

@author: Yang
"""
import multiprocessing,Queue
from sklearn.cross_validation import KFold, StratifiedKFold
import xgboost as xgb
from STFIWF import TfidfVectorizer
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression,RidgeClassifier,PassiveAggressiveClassifier,Lasso,HuberRegressor
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import VotingClassifier,RandomForestClassifier,gradient_boosting
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler
import lightgbm 
from sklearn.ensemble import GradientBoostingClassifier
class term(object):
    def __init__(self):
        random_rate = 8240
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
        #####################################
        clf_svc = SVC(C=1,random_state=random_rate,cache_size=1000)

        self.base_models = basemodel
        self.LR=clf4
        self.svc = clf_svc

    def stacking(self,X,Y,T,wv_X,wv_T,kind):
        """
        ensemble model:stacking
        """
        print ('fitting..')
        models = self.base_models
        folds = list(KFold(len(Y), n_folds=5, random_state=0))
        S_train = np.zeros((X.shape[0], len(models)))
        S_test = np.zeros((T.shape[0], len(models)))

        for i, bm in enumerate(models):
            clf = bm[1]

            S_test_i = np.zeros((T.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = Y[train_idx]
                X_holdout = X[test_idx]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_test[:, i] = S_test_i.mean(1)

        print (S_train.shape,S_test.shape)

        S_train = np.concatenate((S_train,wv_X),axis=1)
        S_test = np.concatenate((S_test, wv_T), axis=1)

        print (S_train.shape,S_test.shape)

        print ('scalering..')
        min_max_scaler = StandardScaler()
        S_train = min_max_scaler.fit_transform(S_train)
        S_test = min_max_scaler.fit_transform(S_test)
        print ('scalering over!')
        self.svc.fit(S_train, Y)
        yp= self.svc.predict(S_test)[:]
        return yp

    def validation(self, X, Y, wv_X, kind):
        """
        2-fold validation
        :param X: train text
        :param Y: train label
        :param wv_X: train wv_vec
        :param kind: age/gender/education
        :return: mean score of 2-fold validation
        """
        print ('向量化中...')
        X=np.array(X)
        fold_n=2
        folds = list(StratifiedKFold(Y, n_folds=fold_n, shuffle=False,random_state=0))
        score = np.zeros(fold_n)
        for j, (train_idx, test_idx) in enumerate(folds):
            print (j+1,'-fold')

            X_train = X[train_idx]
            y_train = Y[train_idx]
            X_test = X[test_idx]
            y_test = Y[test_idx]

            wv_X_train =wv_X[train_idx]
            wv_X_test = wv_X[test_idx]

            vec = TfidfVectorizer(use_idf=True,sublinear_tf=False, max_features=50000, binary=True)
            vec.fit(X_train, y_train)
            X_train = vec.transform(X_train)
            X_test = vec.transform(X_test)

            print ('shape',X_train.shape)

            ypre = self.stacking(X_train,y_train,X_test,wv_X_train,wv_X_test,kind)
            cur = sum(y_test == ypre) * 1.0 / len(ypre)
            score[j] = cur

        print (score)
        print (score.mean(),kind)
        return score.mean()

    def predict(self,X,Y,T,wv_X,wv_T,kind):
        """
        train and predict
        :param X: train text
        :param Y: train label
        :param T: test text
        :param wv_X: train wv
        :param wv_T: test wv
        :param kind: age/gender/education
        :return: array like ,predict of "kind"
        """
        print ('predicting..向量化中...')
        vec = TfidfVectorizer(use_idf=True, sublinear_tf=False, max_features=60000, binary=True)

        vec.fit(X, Y)
        X = vec.transform(X)
        T = vec.transform(T)

        print ('train size',X.shape,T.shape)
        res = self.stacking(X, Y, T, wv_X, wv_T, kind)
        return res
