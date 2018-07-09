# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:59:07 2018

@author: zakis
"""

import pandas as pd
import numpy as np

tags = pd.read_csv('tags_cleaned.csv')
tags = tags.drop('Unnamed: 0', 1)
BOW = pd.read_csv('BOW_final.csv')
BOW = BOW.drop('Unnamed: 0', 1)

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
multilabel_binarizer = MultiLabelBinarizer().fit(tags.values.astype(str))
y = multilabel_binarizer.transform(tags.values.astype(str))
#removing nan
y = np.delete(y, 29, 1) 
#create test and train set
X_train, X_test, y_train, y_test = train_test_split(BOW.values, y, test_size=0.2, random_state=0)

#runnning random forest classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=50)
forest = forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)

#running logistic regression classifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.linear_model import LogisticRegression
classifier = BinaryRelevance(LogisticRegression(n_jobs=-1))
classifier.fit(X_train, y_train)
y_pred_lr = classifier.predict(X_test)

#running extra trees classifier
from sklearn.ensemble import ExtraTreesClassifier
extra_trees = ExtraTreesClassifier(n_estimators=50)
extra_trees = extra_trees.fit(X_train, y_train)
y_pred_extra_trees = extra_trees.predict(X_test)

#choosing the solution by priority (1.extra trees, 2.random forest, 3.logistic regression)
y_pred = []
for i in range(0, len(y_pred_extra_trees)):
    if sum(y_pred_extra_trees[i,:]) == 0:
        if sum(y_pred_lr[i,:]) == 0:
            y_pred.append(y_pred_forest[i,:])
        else:
            y_pred.append(y_pred_lr[i,:])
    else:
        y_pred.append(y_pred_extra_trees[i,:])
y_pred = np.array(y_pred)
            