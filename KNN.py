# -*- coding: utf-8 -*-
"""
Created on Tue May 31 19:52:47 2016

@author: lu
"""

#KNN document classification, using pLSA LSA to evaluate similarity between two documents
import os
import os.path
import re
import random
import math
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import numpy as np

import pickle
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
class Corpus:
    def __init__(self):
        self.WordDF={}
        self.ValidWord={}
        self.english_stopwords=stopwords.words('english')
        #self.stemmer=PorterStemmer()
        self.stemmer=SnowballStemmer('english')
        self.WordIndex={}
        self.docs=[]
        self.label=[]
        self.lsa=None
        self.usvt=None
    def split(self,content):
        words=re.findall('[a-z]+',content)
        ret=[self.stemmer.stem(word) for word in words]
        return ret
        
    def loadFile(self,filename,label):
        myfile=open(filename,'r')
        filecontent=myfile.read()
        myfile.close()
        filecontent=filecontent.lower()
        #words=re.findall('[a-z]+',filecontent)
        words=self.split(filecontent)
        self.docs.append(words)
        self.label.append(label)
        tfdict={}
        for word in words:
            if not self.WordDF.has_key(word):
                self.WordDF[word]=0
            if not tfdict.has_key(word):
                self.WordDF[word]+=1
            tfdict[word]=1
            
    def process(self):
        docs=[]
        for words in self.docs:
            for word in words:
                if self.WordDF[word]>20 and self.WordDF[word]<4000:
                    self.ValidWord[word]=1
        count=0
        for word in self.ValidWord:
            self.WordIndex[word]=count
            count+=1
        for words in self.docs:
            vec=[0]*len(self.WordIndex)
            for word in words:
                if not self.WordIndex.has_key(word):
                    continue
                vec[self.WordIndex[word]]+=1
            for word in self.WordIndex:
                if vec[self.WordIndex[word]]:
                    vec[self.WordIndex[word]]*=(1.0/self.WordDF[word])
            docs.append(vec)
        self.docs=docs
    
    
    def LSA(self):
        row=[]
        col=[]
        data=[]
#        for doc in self.docs:
        for i in range(len(self.docs)):
            doc=self.docs[i]
            _sum=sum(doc)
            for j in range(len(doc)):
                if doc[j]:
                    row.append(i)
                    col.append(j)
                    #normalize
                    data.append(float(doc[j])/_sum)
        coo=coo_matrix((data,(row,col)),shape=(len(self.docs),len(doc)))
        u,s,vt=svds(coo,50)
        self.usvt=(u,s,vt)
        print type(u),type(s),type(vt)
        print u.shape,s.shape,vt.shape
        print s
        
        
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
        
    def lsasim(self,a,b):
        
        return 0.
        
    def cos(self,a,b):
        tmp1=0.
        tmp2=0.
        tmp3=0.
        for i in range(len(a)):
            tmp1+=a[i]*b[i]
            tmp2+=a[i]*a[i]
            tmp3+=b[i]*b[i]
        return tmp1/(math.sqrt(tmp2)*math.sqrt(tmp3))
    def knn_predict(self,filename,K):
        sample=self.doc2vec(filename)
        sims=[]
        #for doc in self.docs:
        for i in range(len(self.docs)):
            doc=self.docs[i]
            label=self.label[i]
            sim=self.cos(doc,sample)
            sims.append((sim,label))
        sims.sort(key=lambda x:x[0],reverse=True)
        cntdict={}
        ret=None
        maxvote=0
        for i in range(K):
            sim,label=sims[i]
            if not cntdict.has_key(label):
                cntdict[label]=0
            cntdict[label]+=1
            if cntdict[label]>maxvote:
                maxvote=cntdict[label]
                ret=label
        return ret
        
def run():
    corp=Corpus()
    subdir=os.listdir('20_newsgroups')
    total=set()
    for dirname in subdir:
        mydir='./20_newsgroups/'+dirname
        files=os.listdir(mydir)
        for filename in files:
            total.add((mydir+'/'+filename,mydir))
    LEN=int(len(total)*0.94)
    train=set(random.sample(total,LEN))
    test=total.difference(train)
    labelmap={}
    labelcnt=0
    for filename,label in train:
    #for filename,label in total:
        corp.loadFile(filename,label)
        if not labelmap.has_key(label):
            labelmap[label]=labelcnt
            labelcnt+=1
    corp.process()
    #print "LSA begin"
    #corp.LSA()
    #return
    
    right=0
    wrong=0    
    for filename,label in test:
        pred=corp.knn_predict(filename,5)
        print label,pred
        if label==pred:
            right+=1
        else:
            wrong+=1
    print right,wrong
run()

