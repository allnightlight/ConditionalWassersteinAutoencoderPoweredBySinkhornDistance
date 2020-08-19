'''
Created on 2020/08/02

@author: ukai
'''
import numpy as np
from wae_environment import WaeEnvironment

class ConcEnvironmentCs02a(WaeEnvironment):
    '''
    classdocs
    '''


    def __init__(self, nBatch, seed = 0, nSample = 2**10):
        WaeEnvironment.__init__(self, nBatch)
        
        self.nX = 3
        self.nZ = 1

        self.randomState = np.random.RandomState(seed=seed)
        self.nSample = nSample
        
    def loadData(self):
        nSample = self.nSample

        def f(r, t):
            x = (r * np.cos(t) + 2) * np.cos(2*t)
            y = (r * np.cos(t) + 2) * np.sin(2*t)
            z = r * np.sin(t)
            X = np.stack((x,y,z), axis=-1) # (*, 3)
            return X
        r = self.randomState.rand(nSample) * 2 - 1
        t = self.randomState.rand(nSample) * np.pi 
        
        X = f(r, t)
        Z = np.zeros((nSample, self.nZ))
        Z[:,0] = 1
        
        self.dataX = X
        self.dataZ = Z


        