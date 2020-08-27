'''
Created on 2020/08/02

@author: ukai
'''
import os
import shutil
import unittest

from builder import Builder
from conc_agent_factory import ConcAgentFactory
from conc_build_parameter import ConcBuildParameter
from conc_environment_factory import ConcEnvironmentFactory
from loader import Loader
from mylogger import MyLogger
from store import Store
from wae_agent import WaeAgent
from wae_trainer_factory import WaeTrainerFactory
from conc_build_parameter_factory import ConcBuildParameterFactory


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
        
        agentFactory = ConcAgentFactory()
        environmentFactory = ConcEnvironmentFactory()
        trainerFactory = WaeTrainerFactory()        
        buildParameterFactory = ConcBuildParameterFactory()
        store = Store(self.dbPath)
        logger = MyLogger(console_print=True)
        
        self.builder = Builder(trainerFactory, agentFactory, environmentFactory, store, logger)

        nIntervalSave = 1
        nEpoch = 2
        
        self.buildParameters = []
        for k1 in range(2):
            self.buildParameters.append(ConcBuildParameter(int(nIntervalSave), int(nEpoch), label="test" + str(k1)))
             
        for k1 in range(2):
            self.buildParameters.append(ConcBuildParameter(int(nIntervalSave), int(nEpoch)
                                                           , label="test case study 02a " + str(k1)
                                                           , nXi = 3
                                                           , target_casestudy = "cs02a"
                                                           ))

        for k1 in range(2):
            self.buildParameters.append(ConcBuildParameter(int(nIntervalSave), int(nEpoch)
                                                           , label="test case study 03a " + str(k1)
                                                           , nXi = 3
                                                           , target_casestudy = "cs03a"
                                                           ))

        for k1 in range(2):
            self.buildParameters.append(ConcBuildParameter(int(nIntervalSave), int(nEpoch)
                                                           , label="test case study 03b " + str(k1)
                                                           , nXi = 3
                                                           , target_casestudy = "cs03b"
                                                           ))

        for k1 in range(2):
            self.buildParameters.append(ConcBuildParameter(int(nIntervalSave), int(nEpoch)
                                                           , label="test case study 03c " + str(k1)
                                                           , nXi = 3
                                                           , target_casestudy = "cs03c"
                                                           ))
        
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
            assert isinstance(buildParameter, ConcBuildParameter)
            self.builder.build(buildParameter)
            
        assert isinstance(self.loader, Loader)        
        for agent, buildParameter, epoch in self.loader.load("test%", None):
            assert isinstance(agent, WaeAgent)
            assert isinstance(buildParameter, ConcBuildParameter)




if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()