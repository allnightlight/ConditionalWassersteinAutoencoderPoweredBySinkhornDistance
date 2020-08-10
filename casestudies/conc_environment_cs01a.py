'''
Created on 2020/08/02

@author: ukai
'''
import numpy as np
from wae_environment import WaeEnvironment

class ConcEnvironmentCs01a(WaeEnvironment):
    '''
    classdocs
    '''


    def __init__(self, nBatch, d_out = 2, d_in = 1, seed = 0, nSample = 2**10):
        WaeEnvironment.__init__(self, nBatch)
        
        self.nX = 2
        self.nZ = 1

        self.d_out = d_out
        self.d_in = d_in
        self.randomState = np.random.RandomState(seed=seed)
        self.nSample = nSample
        
    def loadData(self):
        nSample = self.nSample
        d_out = self.d_out
        d_in = self.d_in

        f = lambda y: np.exp(y * np.log(d_out/d_in)) * d_in/2
        y = self.randomState.rand(nSample)
        r = f(y)
        theta = self.randomState.rand(nSample) * np.pi * 2
        X = np.stack((r * np.cos(theta), r * np.sin(theta)), axis=1) # (*,2)

        Z = np.zeros((nSample, self.nZ))
        Z[:,0] = 1
        
        self.dataX = X
        self.dataZ = Z


