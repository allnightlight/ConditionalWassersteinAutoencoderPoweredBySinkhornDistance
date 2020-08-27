'''
Created on 2020/08/27

@author: ukai
'''

import torch

import numpy as np
from wae_batch_data_environment import WaeBatchDataEnvironment
from wae_environment import WaeEnvironment


class ConcEnvironmentCs03c(WaeEnvironment):
    '''
    classdocs
    '''


    def __init__(self, nBatch, seed = 0, nSample = 2**11, nSampleTest = 2**9):
        WaeEnvironment.__init__(self, nBatch)
        
        self.nX = 3
        self.nZ = 1

        self.randomState = np.random.RandomState(seed=seed)
        self.nSample = nSample
        self.nSampleTest = nSampleTest
        
    def loadData(self):
        nSample = self.nSample
        
        def generateData(nSample):
            
            def generate_trefoil_knot(theta):
                # theta: (...)
                r = 2 + np.cos(3*theta) # (...)
                x = r * np.cos(2*theta) # (...)
                y = r * np.sin(2*theta) # (...)
                z = np.sin(3*theta) # (...)
                
                X = np.stack((x,y,z), axis=-1) # (..., 3)
                return X # (..., 3)
            
            theta = self.randomState.rand(nSample) * 2 * np.pi
            
            X = generate_trefoil_knot(theta)
            
            Z = np.zeros((nSample, self.nZ)) # (nSample, 2)
            Z[:,0] = 1
            
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
