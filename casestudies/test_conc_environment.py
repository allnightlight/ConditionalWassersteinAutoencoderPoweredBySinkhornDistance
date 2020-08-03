'''
Created on 2020/08/02

@author: ukai
'''
import numpy as np
import unittest
from conc_environment_cs01a import ConcEnvironmentCs01a
from builtins import isinstance


class Test(unittest.TestCase):


    def test001(self):
        
        nBatch = 2**5
        d_out = 2
        d_in = 1
        
        environment = ConcEnvironmentCs01a(nBatch, d_out, d_in)
        assert isinstance(environment, ConcEnvironmentCs01a)
        
        environment.loadData()
        
        for batchDataEnvironment in environment.generateBatchDataIterator():
            X = batchDataEnvironment._X.data.numpy() # (*, nX)
            Z = batchDataEnvironment._Z.data.numpy() # (*, nZ
            assert X.shape == (nBatch, environment.nX)
            assert Z.shape == (nBatch, environment.nZ)
            
            r = np.sqrt(np.sum(X**2, axis = -1))
            assert np.all(r > d_in/2-1e-8)
            assert np.all(r < d_out/2+1e-8)
        
            assert np.all(Z[:,0] == 1)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()