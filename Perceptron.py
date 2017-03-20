# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 20:24:24 2016

@author: lu
"""

from sklearn import datasets
import numpy as np
import random


def perceptron_fit(X,Y):
    nfeature=len(X[0])
    max_iter=1000
    w=np.random.random(nfeature)
    b=0.5
    learning_rate=0.1
    #stotistic gradient descent
    for iter_ in range(max_iter):
        i=np.random.randint(0,len(X))
        xi=X[i]
        yi=Y[i]
        if yi*(w.dot(xi)+b)<0:
            w=w+learning_rate*yi*xi
            b=b+learning_rate*yi
    return w,b
            
def run():
    iris=datasets.load_iris()
    X=iris.data
    Y=iris.target
    label=np.unique(Y)
    predictor={}
    #X=X/np.max(X,axis=0)    
    
    index=set(range(len(X)))
    traini=random.sample(index,int(len(X)*0.8))
    testi =list(index.difference(traini))
    
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
        ny=[-1]*len(nx)
        samplex=np.vstack((px,nx))
        sampley=np.array(py+ny)
        
        w,b=perceptron_fit(samplex,sampley)
        predictor[y]=(w,b)
        
    right=0.
    wrong=0.
    for i in range(len(testx)):
        tx=testx[i]
        ty=testy[i]
        maxy=-10000000.
        pred=None
        for label in predictor:
            w,b=predictor[label]            
            y=w.dot(tx)+b
            if y>maxy:
                maxy=y
                pred=label
            
            
        if pred==ty:
            right+=1
        else:
            wrong+=1
        print ty,pred
    print right,wrong,right/(right+wrong)
        
run()