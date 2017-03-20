# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:33:11 2016

@author: lu
"""

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import random
import math
import numpy as np
def test_iris():
    iris=datasets.load_iris()
    X=iris.data
    Y=iris.target
   
    index=set(range(len(X)))
    traini=random.sample(index,int(len(X)*0.8))
    testi =list(index.difference(traini))
    
    intercept=np.array([1]*len(X))
    intercept=np.reshape(intercept,(len(intercept),1))
    X=np.append(X,intercept,1)
    
    trainX=X[traini]
    trainY=Y[traini]
    testX=X[testi]
    testY=Y[testi]    
    
    lr=LogisticRegression()
    lr.fit(trainX,trainY)
    pred=lr.predict(testX)
    right=0.
    wrong=0.
    for i in range(len(pred)):
        if pred[i]==testY[i]:
            right+=1.
        else:
            wrong+=1.
    print right,wrong,right/(right+wrong)    
#test_iris()

def sigmoid(x):
    #print math.e**x/(1+math.e**x)
    return math.exp(x)/(1+math.exp(x))
def log_loss(X,y,w):
    loss=0.
    w=np.array(w)
    for i in range(len(X)):
        x=np.array(X[i])
        wxi=w.dot(x)
        loss+=(y[i]*wxi-math.log(1+math.e**wxi))
    loss*=(-1./len(X))
    return loss
    
def lr_fit(X,y):
    learning_rate=0.25
    max_iter=2000
    nsample=len(X)
    nfeature=len(X[1])
    w=np.random.random(nfeature)
    #w=[0.5]*nfeature
    #print 'Loss:',log_loss(X,y,w)
    prev_loss=log_loss(X,y,w)
    for iter_ in range(max_iter):
        grad=[0.]*nfeature
        for i in range(nfeature):
            for j in range(nsample):
                grad[i]+=(X[j][i]*(y[j]-sigmoid(w.dot(X[j]))))
            grad[i]*=(-1./nsample)
        
        for i in range(nfeature):
            w[i]-=(grad[i]*learning_rate)
        loss=log_loss(X,y,w)
        if loss>prev_loss:
            break
        prev_loss=loss
        #print loss
    #print loss
    return w
def lr_predict(x,w):
    x=np.array(x)
    w=np.array(w)
    return sigmoid(x.dot(w))
        
def run():
    iris=datasets.load_iris()
    X=iris.data
    Y=iris.target
    label=np.unique(Y)
    predictor={}
    #print X[:9,]
    tmp=X/np.max(X,axis=0)
    #print tmp[:9,]
    X=tmp
    #return
    #X=X/np.max(X,axis=0)    
    index=set(range(len(X)))
    traini=random.sample(index,int(len(X)*0.8))
    testi =list(index.difference(traini))
    
    #intercept=np.array([1]*len(X))
    #intercept=np.reshape(intercept,(len(intercept),1))
    #X=np.append(X,intercept,1)
    
    trainx=X[traini]
    trainy=Y[traini]
    testx=X[testi]
    testy=Y[testi]
    
    
    for y in label:
        #positive sample
        px=trainx[trainy==y]
        py=[1]*len(px) 
        #negative sample
        nx=trainx[trainy!=y]
        ny=[0]*len(nx)
        
        samplex=np.vstack((px,nx))
        sampley=np.array(py+ny)
        
        predictor[y]=lr_fit(samplex,sampley)
    
    right=0.
    wrong=0.
    for i in range(len(testx)):
        tx=testx[i]
        ty=testy[i]
        maxposs=0.
        predy=None
        for label in predictor:
            poss=lr_predict(tx,predictor[label])
            if poss>maxposs:
                maxposs=poss
                predy=label
        if predy==ty:
            right+=1
        else:
            wrong+=1
        #print ty,predy
    print right,wrong,right/(right+wrong)
    
run()
    
