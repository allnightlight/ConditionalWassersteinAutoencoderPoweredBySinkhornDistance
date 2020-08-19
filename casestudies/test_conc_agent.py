'''
Created on 2020/08/02

@author: ukai
'''
import unittest

from conc_agent_cs01a import ConcAgentCs01a
from conc_agent_cs02a import ConcAgentCs02a
from conc_environment_cs01a import ConcEnvironmentCs01a
from conc_environment_cs02a import ConcEnvironmentCs02a
from wae_batch_data_agent import WaeBatchDataAgent


class Test(unittest.TestCase):


    def test001(self):
        
        nX, nZ, nH, nXi, nLayer, cluster_interval = (2, 1, 2**3, 2, 2, 3)
        
        agent = ConcAgentCs01a(nX, nZ, nH, nXi, nLayer, cluster_interval)
        
        assert isinstance(agent, ConcAgentCs01a)
        
        nBatch = 2**5
        d_out = 2
        d_in = 1
        
        environment = ConcEnvironmentCs01a(nBatch, d_out, d_in)
        assert isinstance(environment, ConcEnvironmentCs01a)
        
        environment.loadData()
        
        for batchDataEnvironment in environment.generateBatchDataIterator():

            batchDataAgent = agent.forward(batchDataEnvironment)
            assert isinstance(batchDataAgent, WaeBatchDataAgent)
        

    def test002(self):
        
        nBatch = 2**5
        
        environment = ConcEnvironmentCs02a(nBatch)
        assert isinstance(environment, ConcEnvironmentCs02a)
        
        environment.loadData()
        
        nX, nZ, nH, nXi, nLayer, cluster_interval = (environment.nX, environment.nZ, 3, 3, 2, 3)
        
        agent = ConcAgentCs02a(nX, nZ, nH, nXi, nLayer, cluster_interval)
        
        assert isinstance(agent, ConcAgentCs02a)
        
        for batchDataEnvironment in environment.generateBatchDataIterator():

            batchDataAgent = agent.forward(batchDataEnvironment)
            assert isinstance(batchDataAgent, WaeBatchDataAgent)

        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()