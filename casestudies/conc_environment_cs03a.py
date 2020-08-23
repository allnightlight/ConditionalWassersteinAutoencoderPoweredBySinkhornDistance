'''
Created on 2020/08/19

@author: ukai
'''

import torch

import numpy as np
from wae_batch_data_environment import WaeBatchDataEnvironment
from wae_environment import WaeEnvironment


class ConcEnvironmentCs03a(WaeEnvironment):
    '''
    classdocs
    '''


    def __init__(self, nBatch, seed = 0, nSample = 2**11, R = 1, r = 0.1, nSampleTest = 2**9):
        WaeEnvironment.__init__(self, nBatch)
        
        self.nX = 3
        self.nZ = 2

        self.randomState = np.random.RandomState(seed=seed)
        self.nSample = nSample
        self.R = R
        self.r = r
        self.nSampleTest = nSampleTest
        
    def loadData(self):
        nSample = self.nSample
        R, r = self.R, self.r
        
        def generateData(nSample):
            th = self.randomState.rand(nSample//2) * 2 * np.pi
            ph = self.randomState.rand(nSample//2) * 2 * np.pi
            
            xi1 = (R + r * np.cos(th)) * np.cos(ph)
            xi2 = (R + r * np.cos(th)) * np.sin(ph)
            xi3 = r * np.sin(th)
            
            X1 = np.stack((xi1, xi2, xi3), axis=-1) # (*, 3)
            X2 = np.stack((R-xi2, xi3, -xi1), axis=-1) # (*, 3)
            
            X = np.concatenate((X1, X2), axis=0) # (nSample, 3)
            
            Z = np.zeros((nSample, self.nZ)) # (nSample, 2)
            Z[:nSample//2,0] = 1
            Z[nSample//2:,1] = 1
            
            return X, Z
        
        X, Z = generateData(nSample)
        
        self.dataX = X
        self.dataZ = Z
        
        Xtest, Ztest = generateData(self.nSampleTest)
        
        self.testDataX = Xtest
        self.testDataZ = Ztest

    def getTestData(self):
        
        X = self.testDataX.astype(np.float32) # (*, nX)
        Z = self.testDataZ.astype(np.float32) # (*, nZ)

        _X = torch.from_numpy(X) # (*, nX)
        _Z = torch.from_numpy(Z) # (*, nZ)
        
        return WaeBatchDataEnvironment(_X, _Z)
