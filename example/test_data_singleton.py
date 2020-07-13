'''
Created on 2020/07/13

@author: ukai
'''
from datetime import datetime
import unittest

from data_singleton import DataSingleton


class Test(unittest.TestCase):


    def test001(self):        
        
        dbFilePath = "testDb.sqlite"
        tags = ("PV0001", "PV0002")
        period = (datetime.strptime("2020/06/13 00:00", "%Y/%m/%d %H:%M"), datetime.strptime("2020/06/16 00:00", "%Y/%m/%d %H:%M" ))
        samplingIntervalMinute = 5
        
        for _ in range(3):
            dataInst = DataSingleton.getInstance(dbFilePath, period, tags, samplingIntervalMinute)
            assert dataInst.data.shape[1] == len(tags)            

        tags = ("PV0002", "PV0001")

        for _ in range(3):
            dataInst = DataSingleton.getInstance(dbFilePath, period, tags, samplingIntervalMinute)            
            assert dataInst.data.shape[1] == len(tags)            

        tags = ("PV0001", "PV0002", "PV0003")

        for _ in range(3):
            dataInst = DataSingleton.getInstance(dbFilePath, period, tags, samplingIntervalMinute)            
            assert dataInst.data.shape[1] == len(tags)            
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()