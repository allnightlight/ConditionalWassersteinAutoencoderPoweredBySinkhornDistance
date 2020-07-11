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
        
        
    def generateBatchDataIterator(self):
        
        eye = np.eye(self.nZ).astype(np.float32) # (nZ, nZ)
        
        for _ in range(10):
            X = np.random.randn(self.nBatch, self.nX).astype(np.float32) # (*, nX)
            Z = eye[np.random.randint(self.nZ, size=(self.nBatch,)),:] # (*, nZ)
            
            _X = torch.from_numpy(X) # (*, nX)
            _Z = torch.from_numpy(Z) # (*, nZ)
            
            yield WaeBatchDataEnvironment(_X, _Z)