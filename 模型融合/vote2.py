# -*- coding: utf-8 -*-
"""
Created on Sat May 20 16:37:44 2017

@author: Yang
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost
from imblearn.over_sampling import SMOTE

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


smote = SMOTE(random_state=42)
X_train,y_train = smote.fit_sample(df,y)

clf1 = DecisionTreeClassifier()
clf2 = RandomForestClassifier(random_state=42)
clf3 = SVC()
clf4 = LogisticRegression()
clf5 = xgboost.XGBClassifier()

###投票法
eclf = VotingClassifier(estimators=[ ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
eclf.fit(df, y)					
pre = eclf.predict(test)

"""
Soft Voting/Majority Rule classifier.

This module contains a Soft Voting/Majority Rule classifier for
classification estimators.

"""

# Authors: Sebastian Raschka <se.raschka@gmail.com>,
#          Gilles Louppe <g.louppe@gmail.com>
#
# Licence: BSD 3 clause

#import numpy as np
#
#from ..base import BaseEstimator
#from ..base import ClassifierMixin
#from ..base import TransformerMixin
#from ..base import clone
#from ..preprocessing import LabelEncoder
#from ..externals import six
#
#
#class VotingClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
#
#
#    def __init__(self, estimators, voting='hard', weights=None):
#
#        self.estimators_ = estimators
#        self.named_estimators = dict(estimators)
#        self.voting = voting
#        self.weights = weights
#
#    def fit(self, X, y):
#
#        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
#            raise NotImplementedError('Multilabel and multi-output'
#                                      ' classification is not supported.')
#
#        if self.voting not in ('soft', 'hard'):
#            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
#                             % self.voting)
#
#        if self.weights and len(self.weights) != len(self.estimators):
#            raise ValueError('Number of classifiers and weights must be equal'
#                             '; got %d weights, %d estimators'
#                             % (len(self.weights), len(self.estimators)))
#
#        self.le_ = LabelEncoder()
#        self.le_.fit(y)
#        self.classes_ = self.le_.classes_
#        self.estimators_ = []
#
#        for name, clf in self.estimators:
#            fitted_clf = clone(clf).fit(X, self.le_.transform(y))
#            self.estimators_.append(fitted_clf)
#
#        return self
#
#    def predict(self, X):
#
#        if self.voting == 'soft':
#            maj = np.argmax(self.predict_proba(X), axis=1)
#
#        else:  # 'hard' voting
#            predictions = self._predict(X)
#            maj = np.apply_along_axis(lambda x:
#                                      np.argmax(np.bincount(x,
#                                                weights=self.weights)),
#                                      axis=1,
#                                      arr=predictions)
#
#        maj = self.le_.inverse_transform(maj)
#
#        return maj
#
#    def _collect_probas(self, X):
#        """Collect results from clf.predict calls. """
#        return np.asarray([clf.predict_proba(X) for clf in self.estimators_])
#
#    def _predict_proba(self, X):
#        """Predict class probabilities for X in 'soft' voting """
#        avg = np.average(self._collect_probas(X), axis=0, weights=self.weights)
#        return avg
#
#    @property
#    def predict_proba(self):
#        """Compute probabilities of possible outcomes for samples in X.
#
#        Parameters
#        ----------
#        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
#            Training vectors, where n_samples is the number of samples and
#            n_features is the number of features.
#
#        Returns
#        ----------
#        avg : array-like, shape = [n_samples, n_classes]
#            Weighted average probability for each class per sample.
#        """
#        if self.voting == 'hard':
#            raise AttributeError("predict_proba is not available when"
#                                 " voting=%r" % self.voting)
#        return self._predict_proba
#
#    def transform(self, X):
#  
#        if self.voting == 'soft':
#            return self._collect_probas(X)
#        else:
#            return self._predict(X)
#
#    def get_params(self, deep=True):
#        """Return estimator parameter names for GridSearch support"""
#        if not deep:
#            return super(VotingClassifier, self).get_params(deep=False)
#        else:
#            out = super(VotingClassifier, self).get_params(deep=False)
#            out.update(self.named_estimators.copy())
#            for name, step in six.iteritems(self.named_estimators):
#                for key, value in six.iteritems(step.get_params(deep=True)):
#                    out['%s__%s' % (name, key)] = value
#            return out
#
#    def _predict(self, X):
#        """Collect results from clf.predict calls. """
#        return np.asarray([clf.predict(X) for clf in self.estimators_]).T

					
					
					
					
