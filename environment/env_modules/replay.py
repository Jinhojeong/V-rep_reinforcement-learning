from numpy import random

class Replay(object):
    
    def __init__(self,size):
        self.size=size
        self.currentPosition=-1
        self.buffer=[]

    def batch(self,size):
        return random.choice(self.buffer)

    def add(self,experience):
        if (self.currentPosition>=self.size-1):
            self.currentPosition = 0
        if (len(self.buffer)>self.size-1):
            self.memory[self.currentPosition]=experience
        else:
            self.memory.append(experience)
        self.currentPosition+=1
    
    def clear(self):
        self.currentPosition=-1
        self.memory=[]