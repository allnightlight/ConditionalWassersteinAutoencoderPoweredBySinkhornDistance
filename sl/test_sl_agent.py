'''
Created on 2020/07/10

@author: ukai
'''
import unittest
from sl_agent import SlAgent
from builtins import isinstance
from batch_data_in import BatchDataIn
from batch_data_out import BatchDataOut


class Test(unittest.TestCase):


    def test001(self):
        
        agent = SlAgent()
        
        assert isinstance(agent, SlAgent)
        
        batchDataIn = BatchDataIn()
        batchDataOut = agent.forward(batchDataIn)
        
        assert isinstance(batchDataOut, BatchDataOut)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()