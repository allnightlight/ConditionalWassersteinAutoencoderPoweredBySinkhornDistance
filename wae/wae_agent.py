'''
Created on 2020/07/11

@author: ukai
'''

from builtins import isinstance
import os

import torch
from torch.nn import Module, Sequential, Linear, ReLU

import numpy as np
from sl_agent import SlAgent
from util import Utils
from wae_batch_data_agent import WaeBatchDataAgent
from wae_batch_data_environment import WaeBatchDataEnvironment


class WaeAgent(SlAgent, Module):
    '''
    classdocs
    '''

    checkPointPath = "./checkpoint"

    def __init__(self, nX, nZ, nH, nXi):
        '''
        Constructor
        '''
        
        super(WaeAgent, self).__init__()
        
        enc = Sequential(
            Linear(nX, nH)
            , ReLU()
            , Linear(nH, nXi)
            )
        
        dec = Sequential(
            Linear(nXi, nH)
            , ReLU()
            , Linear(nH, nX)
            )
        
        self.enc = enc
        self.dec = dec
        self.nZ = nZ
        self.nXi = nXi
        
    def forward(self, batchDataEnvironment):
        
        assert isinstance(batchDataEnvironment, WaeBatchDataEnvironment)
        
        _X = batchDataEnvironment._X # (*, nX)
        _Z = batchDataEnvironment._Z # (*, nX)
        
        _XiHat = self.enc(_X) # (*, nXi)
        _XHat = self.dec(_XiHat) # (*, nX)
        
        _Xi = self.sampleXi(_Z) # (*, nXi)
        
        return WaeBatchDataAgent(_Xi, _XiHat, _XHat)
        
        
    # <<protected>>
    def sampleXi(self, _Z):
        # _Z: (*, nZ)
        
        nBatch = _Z.shape[0]
        
        Z = _Z.data.numpy() # (*, nZ)
        Offset = np.sum(Z * np.arange(self.nZ), axis=-1) # (*,)
        Tmp = np.random.randn(nBatch, self.nXi) # (*, nXi)
        Tmp[:,0] += Offset # (*,)
        
        _Xi = torch.from_numpy(Tmp.astype(np.float32)) # (*, nXi)
        
        return _Xi # (*, nXi)
        
    def createMemento(self):
        
        filename = Utils.generateRandomString(16) + ".pt"
        path = os.path.join(self.checkPointPath, filename)

        if not os.path.exists(self.checkPointPath):
            os.mkdir(self.checkPointPath)        
        torch.save(self.state_dict(), path)
        
        agentMemento = path
        
        return agentMemento
    
    def loadMemento(self, agentMemento):
        
        path = agentMemento
        self.load_state_dict(torch.load(path))
