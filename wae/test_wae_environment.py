'''
Created on 2020/07/11

@author: ukai
'''
import unittest
from wae_environment import WaeEnvironment
from builtins import isinstance
from wae_batch_data_environment import WaeBatchDataEnvironment


class Test(unittest.TestCase):


    def test001(self):
        
        nBatch = 2**5
        
        environment = WaeEnvironment(nBatch)
        environment.loadData()
        for batchDataEnvironment in environment.generateBatchDataIterator():
            assert isinstance(batchDataEnvironment, WaeBatchDataEnvironment)
            _X = batchDataEnvironment._X
            _Z = batchDataEnvironment._Z
            
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()