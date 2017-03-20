# -*- coding: utf-8 -*-
"""
Created on Sat May 28 21:12:10 2016

@author: Admin
"""
import math
import re
from nltk.stem.snowball import SnowballStemmer
import os
import numpy as np
class PLSA:
    def __init__(self,K=3,max_iter=10):
        self.p_w_z=None#P(wj|zk)
        self.p_z_d=None#P(zk|di)
        self.p_w_z_old=None
        self.p_z_d_old=None
        
        #self.p_z_dw={}
        self.K=K
        self.max_iter=max_iter
        self.N=0
        self.M=0
        self.n={}
        self.stemmer=SnowballStemmer('english')
        self.doccnt=0
        self.wordset=set()
        self.wordindex={}
    
    def init_param(self):
        self.N=self.doccnt
        self.M=len(self.wordset)
        nd=self.doccnt
        nw=len(self.wordset)
        self.p_w_z=np.random.random(size=[nw,self.K])
        self.p_z_d=np.random.random(size=[self.K,nd])
        self.p_w_z_old=np.random.random(size=[nw,self.K])
        self.p_z_d_old=np.random.random(size=[self.K,nd])
        count=0
        for word in self.wordset:
            self.wordindex[count]=word
            count+=1
        
        
        
        for z in range(self.K):
            acc=sum(self.p_w_z[:,z])
            for w in range(nw):
                self.p_w_z[w,z]/=acc
        for d in range(nd):
            acc=sum(self.p_z_d[:,d])
            for z in range(self.K):
                self.p_z_d[z,d]/=acc        
        
        print self.p_w_z
        print self.p_w_z.shape
        print type(self.p_w_z)
        
        
    def split(self,content):
        words=re.findall('[a-z]+',content)
        ret=[self.stemmer.stem(word) for word in words]
        return ret
        
    def loadFile(self,filename,label):
        myfile=open(filename,'r')
        content=myfile.read()
        myfile.close()
        content=content.lower()
        words=self.split(content)
        
        di=self.doccnt
        self.doccnt+=1
        self.n[di]=len(words)
        
        for word in words:
            if not self.n.has_key((di,word)):
                self.n[di,word]=0
            self.n[di,word]+=1
            self.wordset.add(word)
            
    
    def get_p_z_dw(self,zk,di,wj):
        a=self.p_w_z_old[wj,zk]*self.p_z_d_old[zk,di]
        b=0.
        for zi in range(self.K):
            b+=self.p_w_z_old[wj,zi]*self.p_z_d_old[zi,di]
        return a/b
        
    def update_p_w_z(self,wj,zk):
        a=0.
        for di in range(self.N):
            wj_=self.wordindex[wj]
            a+=self.n[di,wj_]*self.get_p_z_dw(zk,di,wj)
        b=0.
        for wm in range(self.M):
            for di in range(self.N):
                wj_=self.wordindex[wj]
                b+=self.n[di,wj_]*self.get_p_z_dw(zk,di,wm)
        #return a/b
        self.p_w_z[wj,zk]=a/b
                
    def update_p_z_d(self,zk,di):
        a=0.
        for wj in range(self.M):
            wj_=self.wordindex[wj]
            a+=self.n[di,wj_]*self.get_p_z_dw(zk,di,wj)
        self.p_z_d[zk,di]=a/self.n[di]

    def log_likelihood(self):
        b=0.
        for di in range(self.N):
            for wj in range(self.M):
                a=0.
                for zk in range(self.K):
                    #a+=self.p_z_dw(zk,di,wj)*math.log(self.p_w_z[wj,zk]*self.p_z_d[zk,di])
                    a+=self.get_p_z_dw(zk,di,wj)*math.log(self.p_w_z[wj,zk]*self.p_z_d[zk,di])
                wj_=self.wordindex[wj]
                b+=(self.n[di,wj_] if self.n.has_key((di,wj_)) else 0) 
                    
        return b
            
    def train(self):
        for istep in range(self.max_iter):
            #E-Step
            for wj in range(self.M):
                for zk in range(self.K):
                    self.p_w_z_old[wj,zk]=self.p_w_z[wj,zk]
            for zk in range(self.K):
                for di in range(self.N):
                    self.p_z_d_old[zk,di]=self.p_z_d_old[zk,di]
                    
            L=self.log_likelihood()
            print "likelyhood",L
            
            #M-Step
            for wj in range(self.M):
                for zk in range(self.K):
                    #update P(wj|zk)
                    self.update_p_w_z(wj,zk)
            for zk in range(self.K):
                for di in range(self.N):
                    #update P(zk|di)
                    self.update_p_z_d(zk,di)
        return 0
def run(): 
    tmp=np.random.random([3,5])
    print tmp
    print tmp[:,0]
    #return
    plsa=PLSA()
    subdir=os.listdir('clustering')
    total=set()
    print subdir
    
    for dirname in subdir:
        mydir='./clustering/'+dirname
        files=os.listdir(mydir)
        for filename in files:
            total.add((mydir+'/'+filename,mydir))
        print mydir
    for filename,label in total:
        plsa.loadFile(filename,label)
        
    plsa.init_param()
    plsa.train()
run()
