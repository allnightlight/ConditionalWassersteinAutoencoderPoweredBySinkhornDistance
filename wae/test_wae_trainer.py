'''
Created on 2020/07/11

@author: ukai
'''
import os
import shutil
import unittest
import numpy as np

from wae_agent import WaeAgent
from wae_environment import WaeEnvironment
from wae_trainer import WaeTrainer


class Test(unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        
        WaeAgent.checkPointPath = "./testCheckPoint"
        
        environment = WaeEnvironment(nBatch = 2**5)
        environment.loadData()
        
        nX = environment.nX
        nZ = environment.nZ
        nH = 2**5
        nXi = 2
        nLayer = int(np.random.choice((1,2)))
        
        agent = WaeAgent(nX, nZ, nH, nXi, nLayer, cluster_interval=3.0, activation="relu")
        
        trainer = WaeTrainer(agent, environment, reg_param = 0.1)
                
        self.environment = environment
        self.agent = agent
        self.trainer = trainer
        
    @classmethod
    def tearDownClass(cls):
        super(Test, cls).tearDownClass()
        
        if os.path.exists(WaeAgent.checkPointPath):
            shutil.rmtree(WaeAgent.checkPointPath)


    def test001(self):
        
        trainer = self.trainer
        assert isinstance(trainer, WaeTrainer)
        
        trainer.train()
        
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()