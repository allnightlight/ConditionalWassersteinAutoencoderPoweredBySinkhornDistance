'''
Created on 2020/08/02

@author: ukai
'''
import numpy as np
from wae_environment import WaeEnvironment

class ConcEnvironmentCs01a(WaeEnvironment):
    '''
    classdocs
    '''


    def __init__(self, nBatch, d_out = 2, d_in = 1):
        WaeEnvironment.__init__(self, nBatch)
        
        self.nX = 2
        self.nZ = 1

        self.d_out = d_out
        self.d_in = d_in
        
    def loadData(self):
        nSample = 2**10
        d_out = self.d_out
        d_in = self.d_in

        f = lambda y: np.exp(y * np.log(d_out/d_in)) * d_in/2
        y = np.random.rand(nSample)
        r = f(y)
        theta = np.random.rand(nSample) * np.pi * 2
        X = np.stack((r * np.cos(theta), r * np.sin(theta)), axis=1) # (*,2)

        Z = np.zeros((nSample, self.nZ))
        Z[:,0] = 1
        
        self.dataX = X
        self.dataZ = Z


