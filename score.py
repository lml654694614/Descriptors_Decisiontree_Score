# -*- coding: utf-8 -*-
"""
Created on Sat May  4 20:57:24 2019

@author: 65469
"""

import pandas as pd
import numpy as np
import os
import time
from sklearn.tree import DecisionTreeClassifier

start=time.time()
os.getcwd()
des=pd.read_csv('descriptors.csv',header=0,nrows=200)
mat=pd.read_csv('trainA.csv',header=0)
matt=pd.read_csv('test.csv',header=0)
score=[]
for d in des.index:
    tau=[]
    taut=[]
    for row in mat.index:
        nA=mat.iloc[row].nA
        nB=mat.iloc[row].nB
        nX=mat.iloc[row].nX
        rA=mat.iloc[row].rA
        rB=mat.iloc[row].rB
        rX=mat.iloc[row].rX
        tau.append(eval(des.descriptor[0]))
    for rowt in matt.index:
        nA=matt.iloc[rowt].nA
        nB=matt.iloc[rowt].nB
        nX=matt.iloc[rowt].nX
        rA=matt.iloc[rowt].rA
        rB=matt.iloc[rowt].rB
        rX=matt.iloc[rowt].rX
        taut.append(eval(des.descriptor[0]))
    X_train=np.array(tau).reshape(len(tau),1)
    X_test=np.array(taut).reshape(len(taut),1)
    y_train=mat.exp_label
    y_test=matt.exp_label
    estimator = DecisionTreeClassifier(max_depth=2)
    estimator.fit(X_train, y_train)
    score.append(estimator.score(X_test,y_test))
des['score']=score
des.to_csv('des_scoreA.csv',index=0,float_format='%.4f')
end=time.time()
print(end-start)
