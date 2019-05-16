# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:04:32 2019

@author: 65469
"""
import numpy as np
import pandas as pd
import os
import time
from sklearn.tree import DecisionTreeClassifier
from multiprocessing.dummy import Pool

def make_score(d):
    tau=[]
    taut=[]
    for row in mat.index:
       nA=float(mat.iloc[row].nA)
       nB=float(mat.iloc[row].nB)
       nX=float(mat.iloc[row].nX)
       rA=mat.iloc[row].rA
       rB=mat.iloc[row].rB
       rX=mat.iloc[row].rX
       tau.append(eval(des.descriptor[d]))
    for rowt in matt.index:
       nA=float(matt.iloc[rowt].nA)
       nB=float(matt.iloc[rowt].nB)
       nX=float(matt.iloc[rowt].nX)
       rA=matt.iloc[rowt].rA
       rB=matt.iloc[rowt].rB
       rX=matt.iloc[rowt].rX
       taut.append(eval(des.descriptor[d]))
    X_train=np.array(tau).reshape(len(tau),1)
    X_test=np.array(taut).reshape(len(taut),1)
    y_train=mat.exp_label
    y_test=matt.exp_label
    estimator = DecisionTreeClassifier(max_depth=2)
    estimator.fit(X_train, y_train)
    score=estimator.score(X_test,y_test)
    return score

start=time.time()
os.getcwd()
des=pd.read_csv('descriptors.csv',header=0,nrows=200)
mat=pd.read_csv('trainA.csv',header=0)
matt=pd.read_csv('test.csv',header=0)
pool = Pool()
# =============================================================================
# deslist=[x for x in range(len(des))]
# =============================================================================
score=pool.map(make_score, range(len(des)))
pool.close()
pool.join()
des['score']=score
des.to_csv('des_score.csv',index=0,float_format='%.4f')
end=time.time()
print(end-start)