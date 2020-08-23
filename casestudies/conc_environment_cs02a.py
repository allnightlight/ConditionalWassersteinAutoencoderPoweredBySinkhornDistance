'''
Created on 2020/08/02

@author: ukai
'''
import torch

import numpy as np
from wae_batch_data_environment import WaeBatchDataEnvironment
from wae_environment import WaeEnvironment


class ConcEnvironmentCs02a(WaeEnvironment):
    '''
    classdocs
    '''


    def __init__(self, nBatch, seed = 0, nSample = 2**10, nSampleTest = 2**9):
        WaeEnvironment.__init__(self, nBatch)
        
        self.nX = 3
        self.nZ = 1

        self.randomState = np.random.RandomState(seed=seed)
        self.nSample = nSample
        self.nSampleTest = nSampleTest
        
    def loadData(self):
        
        def generateData(nSample):
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
            
            return X, Z
        
        X, Z = generateData(self.nSample)
            
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
