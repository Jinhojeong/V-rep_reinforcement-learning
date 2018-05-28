import random
from numpy import arange

class Buffer(object):
    
    def __init__(self,size):
        self.size=size
        self.currentPosition=-1
        self.state0=[]
        self.state1=[]
        self.reward=[]
        self.action=[]
        self.done=[]

    def batch(self,size):
        indices=random.sample(arange(len(self.state0)),min(size,len(self.state0)))
        Batch = []
        for idx in indices:
            Batch.append([self.state0[idx],self.action[idx],self.reward[idx],self.state1[idx]])
        return Batch

    def add(self,element):
        if (self.currentPosition>=self.size-1):
            self.currentPosition = 0
        if (len(self.memory)>self.size-1):
            self.memory[self.currentPosition]=element
        else :
            self.memory.append(element)
        self.currentPosition+=1
    
    def clear(self):
        self.currentPosition=-1
        self.memory=[]