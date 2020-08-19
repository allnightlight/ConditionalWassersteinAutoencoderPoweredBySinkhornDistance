'''
Created on 2020/08/19

@author: ukai
'''

import numpy as np
from wae_environment import WaeEnvironment

class ConcEnvironmentCs03a(WaeEnvironment):
    '''
    classdocs
    '''


    def __init__(self, nBatch, seed = 0, nSample = 2**11, R = 1, r = 0.1):
        WaeEnvironment.__init__(self, nBatch)
        
        self.nX = 3
        self.nZ = 2

        self.randomState = np.random.RandomState(seed=seed)
        self.nSample = nSample
        self.R = R
        self.r = r
        
    def loadData(self):
        nSample = self.nSample
        R, r = self.R, self.r
        
        th = np.random.rand(nSample//2) * 2 * np.pi
        ph = np.random.rand(nSample//2) * 2 * np.pi
        
        xi1 = (R + r * np.cos(th)) * np.cos(ph)
        xi2 = (R + r * np.cos(th)) * np.sin(ph)
        xi3 = r * np.sin(th)
        
        X1 = np.stack((xi1, xi2, xi3), axis=-1) # (*, 3)
        X2 = np.stack((R-xi2, xi3, -xi1), axis=-1) # (*, 3)
        
        X = np.concatenate((X1, X2), axis=0) # (nSample, 3)
        
        Z = np.zeros((nSample, self.nZ)) # (nSample, 2)
        Z[:nSample//2,0] = 1
        Z[nSample//2:,1] = 1
        
        self.dataX = X
        self.dataZ = Z