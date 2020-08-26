'''
Created on 2020/08/19

@author: ukai
'''

import torch
import numpy as np
from wae_agent import WaeAgent

class ConcAgentCs02a(WaeAgent):
    '''
    classdocs
    '''


    def __init__(self, nX, nZ, nH, nXi, nLayer, cluster_interval, activation):
        WaeAgent.__init__(self, nX, nZ, nH, nXi, nLayer, cluster_interval, activation)
        assert nXi == 3
        

    # <<protected>>    
    def sampleXi(self, _Z):
        # _Z: (*, nZ)
        
        nBatch = _Z.shape[0]
        
        th = 2*np.pi*np.random.rand(nBatch) # (*,)
        x = np.cos(th) # (*,)
        y = np.sin(th) # (*,)
        z = np.random.rand(nBatch) # (*,)
        
        Xi = np.stack((x,y,z), axis=-1) # (*,3)
        
        _Xi = torch.from_numpy(Xi.astype(np.float32)) # (*, nXi)
        
        return _Xi # (*, nXi)
        