'''
Created on 2020/07/11

@author: ukai
'''
import os
import shutil
import unittest
import numpy as np

from builder import Builder
from loader import Loader
from mylogger import MyLogger
from store import Store
from wae_agent import WaeAgent
from wae_agent_factory import WaeAgentFactory
from wae_build_parameter import WaeBuildParameter
from wae_build_parameter_factory import WaeBuildParameterFactory
from wae_environment_factory import WaeEnvironmentFactory
from wae_trainer_factory import WaeTrainerFactory


class Test(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        super(Test, cls).setUpClass()
        
        WaeAgent.checkPointPath = "./testCheckPoint"

        cls.dbPath = "testDb.sqlite"
        if os.path.exists(cls.dbPath):
            os.remove(cls.dbPath)
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        
        agentFactory = WaeAgentFactory()
        environmentFactory = WaeEnvironmentFactory()
        trainerFactory = WaeTrainerFactory()        
        buildParameterFactory = WaeBuildParameterFactory()
        store = Store(self.dbPath)
        logger = MyLogger(console_print=True)
        
        self.builder = Builder(trainerFactory, agentFactory, environmentFactory, store, logger)
        
        self.buildParameters = []
        for k1 in range(2):
            nIntervalSave = 10
            nEpoch = 20
            nLayer = int(np.random.choice((1,2)))
            
            self.buildParameters.append(WaeBuildParameter(int(nIntervalSave), int(nEpoch), label="test" + str(k1), nLayer = nLayer))
        
        self.loader = Loader(agentFactory, buildParameterFactory, environmentFactory, store)
        
    @classmethod
    def tearDownClass(cls):
        super(Test, cls).tearDownClass()
        if os.path.exists(cls.dbPath):
            os.remove(cls.dbPath)
                    
        if os.path.exists(WaeAgent.checkPointPath):
            shutil.rmtree(WaeAgent.checkPointPath)


    def test001(self):
        for buildParameter in self.buildParameters:
            assert isinstance(buildParameter, WaeBuildParameter)
            self.builder.build(buildParameter)
            
        assert isinstance(self.loader, Loader)        
        for agent, buildParameter, epoch in self.loader.load("test%", None):
            assert isinstance(agent, WaeAgent)
            assert isinstance(buildParameter, WaeBuildParameter)




if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()