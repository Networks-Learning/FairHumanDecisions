#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:04:11 2018

@author: ivalera
"""

import numpy as np
from load_compas_data import load_compas_data
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import zero_one_loss

def genCases(N, seed):
    np.random.seed(seed)
    # distributions for males and females
    z = np.random.binomial(1, 0.5, size=N)
    pyz=np.zeros(N);
    pyz[z==0]=np.random.beta(3, 5, size=sum(z==0))
    pyz[z==1]=np.random.beta(4, 3, size=sum(z==1))
    
    y = np.random.binomial(1, pyz, size=N)

    return z, y, pyz

def genExperts(V, seed):
    
    np.random.seed(seed)    
    thetaz=np.zeros((V,2));
    thetaz[range(V),0]=np.random.beta(0.5, 0.5, size=V)
    thetaz[range(V),1]=np.random.beta(5, 5, size=V)

    return thetaz

def genExpertsReal(tau1,tau2, pBias,V, seed):

    np.random.seed(seed) 
    #pBias=0.5; #percentaje of bias judges
    bb = np.random.binomial(1, pBias, size=V)
   
    thetaz=np.zeros((V,2));
    thetaz[range(V),0]=np.random.beta(tau1, tau2, size=V)
    thetaz[bb==0,1]=thetaz[bb==0,0]
    thetaz[bb==1,1]=thetaz[bb==1,1]*1.2#(1+np.random.beta(2, 1, size=sum(bb==1)))
    thetaz[thetaz>1.0]=1.0
    
    #print(sum(thetaz[:,1]>=0.5), sum(thetaz[:,0]>=0.5))
    #print(sum(thetaz[:,1]<0.5), sum(thetaz[:,0]<0.5))
    
    return thetaz

def genRealData(pTest,seed):
    x, y, z =load_compas_data()
    y[y==1]=0
    y[y==-1]=1
    z= z['race']
    #
    #x= x[z==0]
    #y= y[z==0]
    #z= z[z==0]
    np.random.seed(seed)
    rs = ShuffleSplit(n_splits=1, test_size=pTest, random_state=0)
    
    idxTrain = None
    idxTest = None
    for idx_Train, idx_Test in rs.split(x):
        idxTrain = idx_Train
        idxTest = idx_Test
        # print("TRAIN:", idxTrain, "TEST:", idxTest)
        
    x_train = x[idxTrain]
    x_test = x[idxTest]
    
    y_train = y[idxTrain]
    y_test = y[idxTest]
    
    z_train = z[idxTrain]
    z_test = z[idxTest]
        
    lr = LR(C=100.0)
    
    #lr = NN(hidden_layer_sizes=(100, 100,100), max_iter=2000)
    #lr = RF(n_estimators=10000)
    p_y_z_test =np.zeros((y_test.size,2))
    
    lr.fit(x_train[z_train==0], y_train[z_train==0])   
    #p_y_z_train = np.exp(lr.predict_log_proba(x_train))
    #print('P(Y|X_train)', p_y_z_train)   
    p_y_z_test[z_test==0] = np.exp(lr.predict_log_proba(x_test[z_test==0]))
    
    lr.fit(x_train[z_train==1], y_train[z_train==1])   
    #p_y_z_train = np.exp(lr.predict_log_proba(x_train))
    #print('P(Y|X_train)', p_y_z_train)   
    p_y_z_test[z_test==1] = np.exp(lr.predict_log_proba(x_test[z_test==1]))
#    print('P(Y|X_test)', p_y_z_test)
#    
    y_train_preds = lr.predict(x_train)
#    y_test_preds = lr.predict(x_test)
#    
    print(1-zero_one_loss(y_train, y_train_preds))
#    
#    print(1-zero_one_loss(y_test, y_test_preds))
    return z_test, y_test, p_y_z_test[:,1] 
