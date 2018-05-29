from numpy import random

class Replay(object):
    
    def __init__(self,size):
        self.size=size
        self.currentPosition=-1
        self.buffer={'state0':[],'state1':[],'action':[],'reward':[],'done':[]}
        self.max=False

    def batch(self,size):
        if len(self.buffer)>size:
            indices=random.choice(range(len(self.buffer)),size,replace=False)
            Batch={'state0':[],'state1':[],'action':[],'reward':[],'done':[]}
            for name in self.buffer.keys():
                for idx in indices:
                    Batch[name].append(self.buffer[name][idx])
            return Batch
        else:
            return self.buffer

    def add(self,experience):
        if (self.currentPosition>=self.size-1):
            self.currentPosition=0
            self.max=True
        if self.max:
            for name in self.buffer.keys():
                self.buffer[name][self.currentPosition]=experience[name]
        else:
            for name in experience.keys():
                self.buffer[name].append(experience[name])
        self.currentPosition+=1
    
    def clear(self):
        self.currentPosition=-1
        self.buffersize=0
        self.buffer={'state0':[],'state1':[],'action':[],'reward':[],'done':[]}
        self.max=False