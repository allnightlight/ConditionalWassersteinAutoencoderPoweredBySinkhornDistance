'''
Created on 2020/07/11

@author: ukai
'''

import torch
import numpy as np

from sl_environment import SlEnvironment
from wae_batch_data_environment import WaeBatchDataEnvironment

class WaeEnvironment(SlEnvironment):
    '''
    classdocs
    '''

    nX = 5
    nZ = 2**1

    def __init__(self, nBatch):
        '''
        Constructor
        '''
        super(WaeEnvironment, self).__init__()
        self.nBatch = nBatch
        self.dataX = None # protected
        self.dataZ = None # protected
        self.loadData()
    
    # <<public>>    
    def generateBatchDataIterator(self):
            
        for idx in self.generateIdxIterator():

            X = self.dataX[idx,:].astype(np.float32) # (*, nX)
            Z = self.dataZ[idx,:].astype(np.float32) # (*, nZ)

            _X = torch.from_numpy(X) # (*, nX)
            _Z = torch.from_numpy(Z) # (*, nZ)
            
            yield WaeBatchDataEnvironment(_X, _Z)
            
    # <<procted>>
    def loadData(self):
        nSample = 2**7
        self.dataX = np.random.randn(nSample, self.nX) # (nSample, nX)
        self.dataZ = np.eye(self.nZ)[np.random.randint(self.nZ, size=(nSample,)), :] # (nSample, nZ), private
    
    # <<procted>>    
    def generateIdxIterator(self):
        nSample = self.dataX.shape[0]
        for _ in range(nSample//self.nBatch):
            idx = np.random.randint(nSample, size=(self.nBatch))
            yield idx
        