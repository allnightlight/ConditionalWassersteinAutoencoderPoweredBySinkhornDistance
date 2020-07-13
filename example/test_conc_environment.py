'''
Created on 2020/07/13

@author: ukai
'''
import unittest

from conc_batch_data_environment import ConcBatchDataEnvironment
from conc_environment import ConcEnvironment
import numpy as np
from wae_batch_data_environment import WaeBatchDataEnvironment


class Test(unittest.TestCase):


    def test001(self):
        
        dbFilePath = "testDb.sqlite"
        ev_tag = "EV0001"
        pv_tags = ("PV0001", "PV0002", )
        period_train_str = ("2020/06/13 00:00", "2020/06/15 00:00", )
        period_test_str= ("2020/06/15 00:00", "2020/06/16 00:00", )
        samplingIntervalMinute = 5
        nBatch = 2**5
        
        environment = ConcEnvironment(dbFilePath, ev_tag, pv_tags, period_train_str, period_test_str, samplingIntervalMinute, nBatch)
        environment.loadData()
        
        cnt = 0
        for batchDataEnvironment in environment.generateBatchDataIterator():

            assert isinstance(batchDataEnvironment, WaeBatchDataEnvironment)
            
            X = batchDataEnvironment._X.data.numpy()
            Z = batchDataEnvironment._Z.data.numpy()
            
            assert ~np.any(np.isnan(X))
            assert ~np.any(np.isnan(Z))            
            assert np.all(np.sum(Z, axis=0) == nBatch)
            
            cnt += 1
        assert cnt > 0
        
        for batchDataEnvironment in [environment.getDataTrain() 
             , environment.getDataTest()]:

            assert isinstance(batchDataEnvironment, ConcBatchDataEnvironment)
            
            X = batchDataEnvironment._X.data.numpy()
            Z = batchDataEnvironment._Z.data.numpy()
            timestamp = batchDataEnvironment.timestamp
            
            assert ~np.any(np.isnan(X))
            assert ~np.any(np.isnan(Z))
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()