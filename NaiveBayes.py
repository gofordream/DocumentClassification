# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:58:37 2016

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
import seaborn as sns
import matplotlib.pylab as plt

class Corpus:
    def __init__(self):
        self.CWord={}
        self.WordTF={}
        self.CountD={}
        self.ValidWord={}
        self.Labels=set()
        self.english_stopwords=stopwords.words('english')
        #self.stemmer=PorterStemmer()
        self.stemmer=SnowballStemmer('english')
    
    def split(self,content):
        words=re.findall('[a-z]+',content)
        #words=nltk.word_tokenize(content)
        #words=content.split("[^a-zA-Z]")
        #ret=[i for i in words if not i in self.english_stopwords]
        #words=nltk.word_tokenize(content)
        ret=[self.stemmer.stem(word) for word in words]
        #ret=words
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
            
            if not self.WordTF.has_key(word):
                self.WordTF[word]=0
            if not tfdict.has_key(word):
                self.WordTF[word]+=1
            tfdict[word]=1
            
    def process(self):
        for label in self.CWord:
            self.CountD[label]=0
            for word in self.CWord[label]:
                if self.WordTF[word]>50 and self.WordTF[word]<3000:
                    self.ValidWord[word]=1
                    self.CountD[label]+=self.CWord[label][word]
    def probability(self,word,label):
        #return P(W|label)
        if not self.ValidWord.has_key(word):
            print "!!!"
        if not self.CWord[label].has_key(word):
            NW=1.
        else:
            NW=self.CWord[label][word]+1.
        prob=float(NW)/self.CountD[label]+len(self.ValidWord)
        prob=math.log(prob)
        return prob
        
    def predict(self,filename):
        myfile=open(filename,'r')
        content=myfile.read()
        myfile.close()
        content=content.lower()
        #words=re.findall('[a-z]+',content)
        words=self.split(content)
        ret=[]
        for label in self.Labels:
            prob=0.
            for word in words:
                if not self.ValidWord.has_key(word):
                    continue
                prob+=self.probability(word,label)
            ret.append((label,prob))
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
    LEN=int(len(total)*0.9)
    train=set(random.sample(total,LEN))
    test=total.difference(train)
    for filename,label in train:
        corp.loadFile(filename,label)
    corp.process()
    
    right=0
    wrong=0
    rightlabel=[]
    wronglabel=[]
    for filename,label in test:
        ret=corp.predict(filename)
        maxprob=0
        result=''
        for L,P in ret:
            if P>maxprob:
                maxprob=P
                result=L
        if label==result:
            right+=1
            rightlabel.append(label)
        else:
            wrong+=1
            wronglabel.append(label)
    print right,wrong,float(right)/(right+wrong)
    print len(corp.ValidWord)
    #plt.figure(figsize=(20,5))
    #plt.figure(1)
    #sns.countplot(rightlabel)
    #plt.figure(2)
    #sns.countplot(wronglabel)
run()
#nltk.download()


