'''
Created on 2020/07/11

@author: ukai
'''
import os
import shutil
import unittest

import numpy as np
from wae_agent import WaeAgent
from wae_batch_data_environment import WaeBatchDataEnvironment
from wae_environment import WaeEnvironment


class Test(unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        
        WaeAgent.checkPointPath = "./testCheckPoint"
        
        environment = WaeEnvironment(nBatch = 2**5)
        
        nX = environment.nX
        nZ = environment.nZ
        nH = 2**5
        nXi = 2
        
        agent = WaeAgent(nX, nZ, nH, nXi, cluster_interval = 3.0)
        agentAnother = WaeAgent(nX, nZ, nH, nXi, cluster_interval = 3.0)
                
        self.agent = agent
        self.agentAnother = agentAnother
        self.environment = environment
        
    @classmethod
    def tearDownClass(cls):
        super(Test, cls).tearDownClass()
        
        if os.path.exists(WaeAgent.checkPointPath):
            shutil.rmtree(WaeAgent.checkPointPath)


    def test001(self):
        agent = self.agent
        agentAnother = self.agentAnother
        assert isinstance(agent, WaeAgent)
        
        agentMemento = agent.createMemento()
        
        agentAnother.loadMemento(agentMemento)
        
        for _p1, _p2 in zip(agent.parameters(), agentAnother.parameters()):
            p1 = _p1.data.numpy()
            p2 = _p2.data.numpy()
            assert np.all(p1 == p2)        
        
    def test002(self):
        
        agent = self.agent
        assert isinstance(agent, WaeAgent)
        
        for batchDataEnvironment in self.environment.generateBatchDataIterator():
            batchDataAgent = agent(batchDataEnvironment)
            
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()