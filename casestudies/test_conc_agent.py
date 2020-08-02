'''
Created on 2020/08/02

@author: ukai
'''
import unittest

from conc_agent_cs01a import ConcAgentCs01a
from conc_environment_cs01a import ConcEnvironmentCs01a
from wae_batch_data_agent import WaeBatchDataAgent


class Test(unittest.TestCase):


    def test001(self):
        
        nX, nZ, nH, nXi, cluster_interval = (2, 1, 2**3, 2, 3)
        
        agent = ConcAgentCs01a(nX, nZ, nH, nXi, cluster_interval)
        
        isinstance(agent, ConcAgentCs01a)
        
        nBatch = 2**5
        d_out = 2
        d_in = 1
        
        environment = ConcEnvironmentCs01a(nBatch, d_out, d_in)
        assert isinstance(environment, ConcEnvironmentCs01a)
        
        environment.loadData()
        
        for batchDataEnvironment in environment.generateBatchDataIterator():

            batchDataAgent = agent.forward(batchDataEnvironment)
            assert isinstance(batchDataAgent, WaeBatchDataAgent)
        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()