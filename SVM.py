# -*- coding: utf-8 -*-
"""
Created on Fri May 27 22:05:46 2016

@author: lu
"""

import os
import os.path
import re
import random
import math
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

import pickle
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from scipy.sparse import coo_matrix

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
                if self.WordDF[word]>20 and self.WordDF[word]<4000:
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
        
        
def run():
    corp=Corpus()
    subdir=os.listdir('20_newsgroups')
    total=set()
    for dirname in subdir:
        mydir='./20_newsgroups/'+dirname
        files=os.listdir(mydir)
        for filename in files:
            total.add((mydir+'/'+filename,mydir))
    LEN=int(len(total)*0.9)
    train=set(random.sample(total,LEN))
    test=total.difference(train)
    labelmap={}
    labelcnt=0
    for filename,label in train:
        corp.loadFile(filename,label)
        if not labelmap.has_key(label):
            labelmap[label]=labelcnt
            labelcnt+=1
    corp.process()
    
    train_sample=[]
    train_y=[]
    col=[]
    row=[]
    data=[]
    row_index=0
    
    for filename,label in train:
        sample=corp.doc2vec(filename)
        #sample.append(labelmap[label])
        train_y.append(labelmap[label])
        train_sample.append(sample)
        for i in range(len(sample)):
            if sample[i]!=0:
                row.append(row_index)
                col.append(i)
                data.append(sample[i])
        row_index+=1
        
    coo=coo_matrix((data,(row,col)),shape=(row_index,len(sample)))
    #use sparse matrix to reduce training time
        
    #mysvc=SVC()
    mysvc=LinearSVC()
    #mysvc.fit(train_sample,train_y)
    mysvc.fit(coo,train_y)
    
    test_sample=[]
    real=[]
    for filename,label in test:
        sample=corp.doc2vec(filename)
        test_sample.append(sample)
        real.append(labelmap[label])
        
    pred=mysvc.predict(test_sample)
    
    right=0
    wrong=0
    for i in range(len(real)):
        if real[i]==pred[i]:
            right+=1
        else:
            wrong+=1
    print right,wrong,float(right)/(right+wrong)
    
        
    
run()

            