# -*- coding: utf-8 -*-
"""
Created on Sun May 29 10:57:12 2016

@author: Admin
"""

#from sklearn.cluster import KMeans
from sklearn import datasets
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import re
import os

class Corpus:
    def __init__(self):
        self.CWord={}
        self.WordDF={}
        self.CountD={}
        self.ValidWord={}
        self.Labels=set()
        self.english_stopwords=stopwords.words('english')
        #self.stemmer=PorterStemmer()
        self.stemmer=SnowballStemmer('english')
        
        self.WordIndex={}
    
    def split(self,content):
        words=re.findall('[a-z]+',content)
        ret=[self.stemmer.stem(word) for word in words]
        return ret
        
    def loadFile(self,filename,label):
        self.Labels.add(label)
        myfile=open(filename,'r')
        filecontent=myfile.read()
        myfile.close()
        filecontent=filecontent.lower()
        #words=re.findall('[a-z]+',filecontent)
        words=self.split(filecontent)
        tfdict={}
        for word in words:
            if not self.CWord.has_key(label):
                self.CWord[label]={}
            if not self.CWord[label].has_key(word):
                self.CWord[label][word]=0
            self.CWord[label][word]+=1
            
            if not self.WordDF.has_key(word):
                self.WordDF[word]=0
            if not tfdict.has_key(word):
                self.WordDF[word]+=1
            tfdict[word]=1
            
    def process(self):
        for label in self.CWord:
            self.CountD[label]=0
            for word in self.CWord[label]:
                if self.WordDF[word]>20 and self.WordDF[word]<1000:
                    self.ValidWord[word]=1
                    self.CountD[label]+=self.CWord[label][word]
        count=0
        for word in self.ValidWord:
            self.WordIndex[word]=count
            count+=1
    def doc2vec(self,filename):
        myfile=open(filename,'r')
        content=myfile.read()
        myfile.close()
        content=content.lower()
        words=self.split(content)
        sample=[0]*len(self.WordIndex)
        #calculate TF
        for word in words:
            if not self.WordIndex.has_key(word):
                continue
            sample[self.WordIndex[word]]+=1
        #multiplied by IDF
        for word in self.WordIndex:
            if sample[self.WordIndex[word]]:
                sample[self.WordIndex[word]]*=(1.0/self.WordDF[word])
        return sample


def k_means(X,labels,K):
    max_iter=20
    centroids=random.sample(X,K)
    
    for i in range(max_iter):
        cluster={}
        clusterlabel={}
        for i in range(K):
            cluster[i]=[]
            clusterlabel[i]=[]
            
        #for x in X:
        
        for j in range(len(X)):
            x=X[j]
            label=labels[j]
            maxsim=-100.0
            index=-1
            for i in range(K):
                sim=cos(centroids[i],x)
                if sim>maxsim:
                    maxsim=sim
                    index=i
            cluster[index].append(x)
            clusterlabel[index].append(label)
            
        #
        for k_cluster in clusterlabel:
            print 'cluster:',k_cluster
            tmpdict={}
            for label in clusterlabel[k_cluster]:
                if not tmpdict.has_key(label):
                    tmpdict[label]=0
                tmpdict[label]+=1
            for label in tmpdict:
                print label,tmpdict[label]
        
        #update center
        centroids=[]
        acccohe=0.
        for k_cluster in cluster:
            acc=np.array([0.]*len(X[0]))
            for x in cluster[k_cluster]:
                acc+=np.array(x)
            centroid=acc/float(len(cluster[k_cluster]))
            centroids.append(centroid)
            #print centroid
            acccohe+=cohesion(cluster[k_cluster],centroid)
        print acccohe
    X=[]
    y=[]
    for k_cluster in cluster:
        for x in cluster[k_cluster]:
            X.append(x)
            y.append(k_cluster)
    return X,y
        
def l2_dist(a,b):
    acc=0.
    for i in range(len(a)):
        acc+=(a[i]-b[i])**2
    return math.sqrt(acc)
            
def cos(a,b):
    tmp1=0.
    tmp2=0.
    tmp3=0.
    for i in range(len(a)):
        tmp1+=a[i]*b[i]
        tmp2+=a[i]*a[i]
        tmp3+=b[i]*b[i]
    return tmp1/(math.sqrt(tmp2)*math.sqrt(tmp3))
    
def cohesion(cluster,centroid):
    cohe=0.
    for x in cluster:
        cohe+=cos(x,centroid)
    return cohe

def test():
    cluster=[]
    for i in range(20):
        x=np.random.random(5)
        cluster.append(x)
    centroid=[0]*5
    for i in range(5):
        tmp=0.
        for j in range(20):
            tmp+=cluster[j][i]
        centroid[i]=tmp/20
    print cohesion(cluster,centroid)
    for x in cluster:
        print cohesion(cluster,x)
#test()
    
def run0():
    iris=datasets.load_iris()
    #X=iris.data[:,:2]
    X=iris.data
    y=iris.target
    #print type(X)
    #print X-np.mean(X,axis=0)
    #return
    
    X_reduced=PCA(n_components=2).fit_transform(X)
    #print X_reduced
    #plt.scatter(X_reduced[:,0],X_reduced[:,1])
    #plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Paired)
    #print type(X)

    #sns.stripplot(X_reduced[:,0],X_reduced[:,1])    
    #sns.pairplot(X_reduced)
    
    X,y=k_means(X,3)
    
    y=np.array(y)
    X=np.array(X)
    print X.shape
    print X[y==1].shape,X[y==0].shape,X[y==2].shape
    
    X_reduced=PCA(n_components=2).fit_transform(X)
    plt.scatter(X_reduced[y==0,0],X_reduced[y==0,1],marker='x',color='m',label='0',s=50)
    plt.scatter(X_reduced[y==1,0],X_reduced[y==1,1],marker='+',color='c',label='1',s=50)
    plt.scatter(X_reduced[y==2,0],X_reduced[y==2,1],marker='*',color='g',label='2',s=50)
    print X.shape
    #plt.scatter(X[y==0,0],X[y==0,1],marker='x',color='m',label='0',s=50)
    #plt.scatter(X[y==1,0],X[y==1,1],marker='+',color='c',label='1',s=50)
    #plt.scatter(X[y==2,0],X[y==2,1],marker='*',color='g',label='2',s=50)
def run():
    corp=Corpus()
    subdir=os.listdir('clustering')
    dataset=set()
    for dirname in subdir:
        mydir='./clustering/'+dirname
        files=os.listdir(mydir)
        for filename in files:
            dataset.add((mydir+'/'+filename,mydir))
    labelmap={}
    labelcnt=0
    for filename,label in dataset:
        corp.loadFile(filename,label)
        if not labelmap.has_key(label):
            labelmap[label]=labelcnt
            labelcnt+=1
    corp.process()
    
    samplex=[]
    sampley=[]
    col=[]
    row=[]
    data=[]
    row_index=0
    
    for filename,label in dataset:
        sample=corp.doc2vec(filename)
        #sample.append(labelmap[label])
        sampley.append(labelmap[label])
        samplex.append(sample)
        
        """for i in range(len(sample)):
            if sample[i]!=0:
                row.append(row_index)
                col.append(i)
                data.append(sample[i])
        row_index+=1"""
        
    #coo=coo_matrix((data,(row,col)),shape=(row_index,len(sample)))
    print len(samplex),len(samplex[0])

    #X_reduced=PCA(n_components=2).fit_transform(samplex)    
    #print "F"
    
    X,y=k_means(samplex,sampley,3)
    y=np.array(y)
    X=np.array(X)
    X_reduced=PCA(n_components=2).fit_transform(X)    
    plt.scatter(X_reduced[y==0,0],X_reduced[y==0,1],marker='x',color='m',label='0',s=50)
    plt.scatter(X_reduced[y==1,0],X_reduced[y==1,1],marker='+',color='c',label='1',s=50)
    plt.scatter(X_reduced[y==2,0],X_reduced[y==2,1],marker='*',color='g',label='2',s=50)
    #print X_reduced[:100,]
    
run()