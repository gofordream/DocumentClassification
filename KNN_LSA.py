# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 19:28:48 2016

@author: lu
"""

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
from nltk.stem.snowball import SnowballStemmer

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
        #self.label=[]
        self.lsa=None
        #self.usvt=None
        self.doccnt=0
        self.traindoc=[]
        self.testdoc =[]
        self.trainlabel=[]
        self.testlabel =[]
        self.U=None
        
        
    def split(self,content):
        words=re.findall('[a-z]+',content)
        ret=[self.stemmer.stem(word) for word in words]
        return ret
        
    def loadFile(self,filename,label,train):
        if train:
            self.traindoc.append(self.doccnt)
            self.trainlabel.append(label)
        else:
            self.testdoc.append(self.doccnt)
            self.testlabel.append(label)
        self.doccnt+=1
        
        myfile=open(filename,'r')
        filecontent=myfile.read()
        myfile.close()
        filecontent=filecontent.lower()
        #words=re.findall('[a-z]+',filecontent)
        words=self.split(filecontent)
        self.docs.append(words)
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
                if self.WordDF[word]>20 and self.WordDF[word]<10000:
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
#       for doc in self.docs:
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
        u,s,vt=svds(coo,2000)
        #self.usvt=(u,s,vt)
        self.U=u
        print type(u),type(s),type(vt)
        print u.shape,s.shape,vt.shape
        print s
    
    def cos(self,a,b):
        tmp1=0.
        tmp2=0.
        tmp3=0.
        for i in range(len(a)):
            tmp1+=a[i]*b[i]
            tmp2+=a[i]*a[i]
            tmp3+=b[i]*b[i]
        return tmp1/(math.sqrt(tmp2)*math.sqrt(tmp3))
        

    def knn_predict(self,di,K):
        #sample=self.doc2vec(filename)
        sims=[]
        #for i in range(len(self.docs)):
        for i in range(len(self.traindoc)):
            #doc=self.docs[self.traindoc[i]]
            label=self.trainlabel[i]

            index=self.traindoc[i]
            
            #sim=self.cos(doc,sample)
            #sim=self.lsasim(di,traindoc[i])
            c1=self.U[index]
            c2=self.U[di]
            sim=c1.dot(c2)
            sims.append((sim,label))
        sims.sort(key=lambda x:x[0],reverse=True)
        cntdict={}
        ret=None
        maxvote=0
        for i in range(K):
            sim,label=sims[i]
            if not cntdict.has_key(label):
                cntdict[label]=0
            #cntdict[label]+=1
            cntdict[label]+=sim
            if cntdict[label]>maxvote:
                maxvote=cntdict[label]
                ret=label
        return ret
        
    def predict(self):
        right=0
        wrong=0
        for i in range(len(self.testdoc)):
            di=self.testdoc[i]
            label=self.testlabel[i]
            pred=self.knn_predict(di,5)
            print label,pred
            if label==pred:
                right+=1
            else:
                wrong+=1
        print right,wrong,float(right)/(right+wrong)
        return
        
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
    #labelmap={}
    #labelcnt=0
    for filename,label in train:
        corp.loadFile(filename,label,True)
        #if not labelmap.has_key(label):
        #    labelmap[label]=labelcnt
        #    labelcnt+=1
    for filename,label in test:
        corp.loadFile(filename,label,False)
    corp.process()
    print "LSA begin"
    corp.LSA()
    corp.predict()
    print len(corp.ValidWord)
    return
run()

