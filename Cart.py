# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 22:36:51 2016

@author: lu
"""

from sklearn.datasets import load_boston
import random
from sklearn.metrics import mean_squared_error
import numpy as np

def error(data):
    mean=np.mean(data)
    err=0.
    for e in data:
        err+=((e-mean)**2)
    return err

def split(data,root,depth=1,randomFeature=False):
    if len(data)<15 or depth>=10:
        root['value']=np.mean(data[:,-1])
        return
    maxerr=100000000.
    
    nfeature=len(data[0])-1
    bestleft=None
    bestright=None
    splitfea=None
    splitval=None
    subfea=range(nfeature)
    if randomFeature:
        feainx=range(nfeature)
        subfea=random.sample(feainx,int(len(feainx)*0.7))
    #for i in range(nfeature):
    for i in subfea:
        cut=np.unique(data[:,i])
        for cp in cut:
            left =data[data[:,i]<=cp]
            right=data[data[:,i]> cp]
            lerr=error(left[:,-1])
            rerr=error(right[:,-1])
            err=lerr+rerr
            if err<maxerr:
                maxerr=err
                bestleft=left
                bestright=right
                splitfea=i
                splitval=cp
                
    rooterr=error(data[:,-1])
    if maxerr>=rooterr:
        root['value']=np.mean(data[:,-1])
        return
    if bestleft is not None:
        root['splitfea']=splitfea
        root['splitval']=splitval
        root['left']={}
        split(bestleft,root['left'],depth+1)
    if bestright is not None:
        root['splitfea']=splitfea
        root['splitval']=splitval
        root['right']={}
        split(bestright,root['right'],depth+1)

def cart_fit(X,Y,randomFeature=False):
    Y=np.reshape(Y,(len(Y),1))
    data=np.append(X,Y,1)
    root={}
    split(data,root,1,randomFeature)
    return root
    
def cart_predict(X,root):
    pred=[]
    for x in X:
        node=root
        while True:
            splitfea=node['splitfea'] if node.has_key('splitfea') else None
            splitval=node['splitval'] if node.has_key('splitval') else None
            if splitfea is not None:
                if x[splitfea]<=splitval:
                    node=node['left']
                else:
                    node=node['right']
            else:
                pred.append(node['value'])
                break
            
    return pred
def cart_bagging_fit(n_estimators,X,Y,randomFeature=False):
    estimators=[]
    for i in range(n_estimators):
        si=np.random.randint(0,len(X),len(X))
        XI=X[si]
        YI=Y[si]
        root=cart_fit(XI,YI,randomFeature)
        estimators.append(root)
    return estimators
def cart_bagging_predict(X,estimators):
    total=np.array([0.]*len(X))
    for root in estimators:
        pred=cart_predict(X,root)
        total=total+pred
    pred=total/len(estimators)
    return pred


def cart_boosting_predict(X,estimators):
    acc=np.array([0.]*len(X))
    for root in estimators:
        pred=cart_predict(X,root)
        acc=acc+pred
    return acc
def cart_boosting_fit(n_estimators,X,Y):
    estimators=[]
    for i in range(n_estimators):
        pred=cart_boosting_predict(X,estimators)
        ngrad=Y-pred
        root=cart_fit(X,ngrad)
        estimators.append(root)
    return estimators
        
def run():
    boston=load_boston()
    X=boston.data
    Y=boston.target
    
    index=set(range(len(X)))
    traini=random.sample(index,int(len(index)*0.8))
    testi=list(index.difference(traini))
    
    trainx=X[traini]
    trainy=Y[traini]
    testx= X[testi]
    testy= Y[testi]
    
    root=cart_fit(trainx,trainy)
    pred=cart_predict(testx,root)
    print mean_squared_error(testy,pred)**0.5
    
    estimators=cart_bagging_fit(20,trainx,trainy)
    pred=cart_bagging_predict(testx,estimators)
    print mean_squared_error(testy,pred)**0.5
    
    estimators=cart_bagging_fit(20,trainx,trainy,True)
    pred=cart_bagging_predict(testx,estimators)
    print mean_squared_error(testy,pred)**0.5
    
    estimators=cart_boosting_fit(20,trainx,trainy)
    pred=cart_boosting_predict(testx,estimators)
    print mean_squared_error(testy,pred)**0.5
    
run()
    