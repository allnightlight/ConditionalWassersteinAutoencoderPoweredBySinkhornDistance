'''
Created on 2020/08/02

@author: ukai
'''
from builtins import isinstance
import unittest

from conc_environment_cs01a import ConcEnvironmentCs01a
from conc_environment_cs02a import ConcEnvironmentCs02a
from conc_environment_cs03a import ConcEnvironmentCs03a
from conc_environment_cs03b import ConcEnvironmentCs03b
import numpy as np


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


    def test002(self):
        
        nBatch = 2**5
        
        environment = ConcEnvironmentCs02a(nBatch)
        assert isinstance(environment, ConcEnvironmentCs02a)
                
        environment.loadData()
        cach = environment.dataX
        
        environment = ConcEnvironmentCs02a(nBatch)
        environment.loadData() 
        assert np.all(cach == environment.dataX)

        
        for batchDataEnvironment in environment.generateBatchDataIterator():
            X = batchDataEnvironment._X.data.numpy() # (*, nX)
            Z = batchDataEnvironment._Z.data.numpy() # (*, nZ
            assert X.shape == (nBatch, environment.nX)
            assert Z.shape == (nBatch, environment.nZ)            
            assert np.all(Z[:,0] == 1)
            
        batchDataEnvironment = environment.getTestData()
        X = batchDataEnvironment._X.data.numpy() # (*, nX)
        Z = batchDataEnvironment._Z.data.numpy() # (*, nZ
        assert X.shape == (environment.nSampleTest, environment.nX)
        assert Z.shape == (environment.nSampleTest, environment.nZ)            
        assert np.all(Z[:,0] == 1)


    def test003(self):
        
        nBatch = 2**5
        
        environment_a = ConcEnvironmentCs03a(nBatch)
        assert isinstance(environment_a, ConcEnvironmentCs03a)
        
        environment_b = ConcEnvironmentCs03b(nBatch)
        assert isinstance(environment_b, ConcEnvironmentCs03b)
        
        for environment in (environment_a, environment_b):
        
            environment.loadData()
            cach = environment.dataX
            
            environment = ConcEnvironmentCs03a(nBatch)
            environment.loadData() 
            assert np.all(cach == environment.dataX)
            
            for batchDataEnvironment in environment.generateBatchDataIterator():
                X = batchDataEnvironment._X.data.numpy() # (*, nX)
                Z = batchDataEnvironment._Z.data.numpy() # (*, nZ
                assert X.shape == (nBatch, environment.nX)
                assert Z.shape == (nBatch, environment.nZ)            
                assert np.all(Z[:,0] + Z[:,1] == 1)
    
            batchDataEnvironment = environment.getTestData()
            X = batchDataEnvironment._X.data.numpy() # (*, nX)
            Z = batchDataEnvironment._Z.data.numpy() # (*, nZ
            assert X.shape == (environment.nSampleTest, environment.nX)
            assert Z.shape == (environment.nSampleTest, environment.nZ)            
            assert np.all(Z[:,0] + Z[:,1] == 1)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()