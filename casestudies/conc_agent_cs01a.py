'''
Created on 2020/08/02

@author: ukai
'''

import torch
import numpy as np
from wae_agent import WaeAgent

class ConcAgentCs01a(WaeAgent):
    '''
    classdocs
    '''


    def __init__(self, nX, nZ, nH, nXi, cluster_interval):
        WaeAgent.__init__(self, nX, nZ, nH, nXi, cluster_interval)
        

    # <<protected>>    
    def sampleXi(self, _Z):
        # _Z: (*, nZ)
        
        nBatch = _Z.shape[0]
        
        Xi = np.random.randn(nBatch, self.nXi) # (:, nXi)
        
        _Xi = torch.from_numpy(Xi.astype(np.float32)) # (*, nXi)
        
        return _Xi # (*, nXi)
