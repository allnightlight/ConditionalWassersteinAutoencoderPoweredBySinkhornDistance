'''
Created on 2020/08/19

@author: ukai
'''

import torch
import numpy as np
from wae_agent import WaeAgent

class ConcAgentCs03a(WaeAgent):
    '''
    classdocs
    '''


    def __init__(self, nX, nZ, nH, nXi, nLayer, cluster_interval, activation, R = 1.0, r = 0.1):
        WaeAgent.__init__(self, nX, nZ, nH, nXi, nLayer, cluster_interval, activation)
        assert nXi == 3
        assert nZ == 2
        
        self.R = R
        self.r = r
        

    # <<protected>>    
    def sampleXi(self, _Z):
        # _Z: (*, nZ)
        
        nBatch = _Z.shape[0]
        R, r = self.R, self.r
        
        th = np.random.rand(nBatch) * 2 * np.pi
        ph = np.random.rand(nBatch) * 2 * np.pi
                
        xi1 = (R + r * np.cos(th)) * np.cos(ph)
        xi2 = (R + r * np.cos(th)) * np.sin(ph)
        xi3 = r * np.sin(th)
        
        Xi = np.stack((xi1, xi2, xi3), axis=-1) # (*, nXi = 3)

        Z = _Z.data.numpy()
        Xi[Z[:,0]==1,2] += R/2
        Xi[Z[:,1]==1,2] -= R/2
        
        _Xi = torch.from_numpy(Xi.astype(np.float32)) # (*, nXi=3)
        
        return _Xi # (*, nXi=3)
        