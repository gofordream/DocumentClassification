# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 20:15:38 2016

@author: lu
"""

from sklearn.datasets import load_boston
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def lr_fit(X,Y):
    nsample=len(X)
    nfeature=len(X[0])
    w=np.random.random(nfeature)
    
    #pred=lr_predict(X,w)
    #print 'rmse:',mean_squared_error(Y,pred)
    
    max_iter=1000
    learning_rate=0.25
    rmse_prev=100000000.
    for iter_ in range(max_iter):
        grad=[0.]*len(w)
        for j in range(len(grad)):
            for i in range(nsample):
                xi=np.array(X[i])
                yi=Y[i]
                grad[j]+=((yi-w.dot(xi))*(-xi[j]))
            grad[j]*=(1.0/nsample)
            #grad[j]+=0.2*w[j]#regulization
        
        for j in range(len(w)):
            w[j]-=(learning_rate*grad[j])
        
        #print w
        pred=lr_predict(X,w)
        rmse=mean_squared_error(Y,pred)**0.5
        if rmse>rmse_prev:
            print iter_,rmse
            break
        rmse_prev=rmse
    #print w
    return w
def lr_predict(X,w):
    pred=[]
    w=np.array(w)
    for x in X:
        x=np.array(x)
        y=x.dot(w)
        pred.append(y)
    return pred
def run():
    boston=load_boston()
    X=boston.data
    #print np.max(X,axis=0)
    
    X=X/np.max(X,axis=0)
    #print X[:,6]
    y=boston.target
    #print y
    index=set(range(len(X)))
    traini=random.sample(index,int(len(index)*0.8))
    testi =list(index.difference(traini))
    lr=LinearRegression()
    trainx=X[traini]
    trainy=y[traini]
    testx =X[testi]
    testy =y[testi]
    lr.fit(trainx,trainy)
    pred=lr.predict(testx)
    
    #print np.mean(testy)
    print mean_squared_error(testy,pred)**0.5
    
    w=lr_fit(trainx,trainy)
    pred=lr_predict(testx,w)
    print mean_squared_error(testy,pred)**0.5
    
run()
