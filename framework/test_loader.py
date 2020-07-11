'''
Created on 2020/07/10

@author: ukai
'''
import os
import unittest

from agent import Agent
from agent_factory import AgentFactory
from build_parameter import BuildParameter
from build_parameter_factory import BuildParameterFactory
from loader import Loader
from store import Store
from store_field import StoreField
from environment_factory import EnvironmentFactory


class Test(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        super(Test, cls).setUpClass()
        
        dbPath = "testDb.sqlite"
        if os.path.exists(dbPath):
            os.remove(dbPath)
        
        store = Store(dbPath)
        assert isinstance(store, Store)
        
        for k1 in range(2**3):
            buildParameter = BuildParameter(label = "test" + str(k1))
            agent = Agent()

            for epoch in range(2**4):        
        
                agentMemento = agent.createMemento()
                buildParameterMemento = buildParameter.createMemento()
                buildParameterLabel = buildParameter.label
        
                storeField = StoreField(agentMemento, epoch, buildParameterMemento, buildParameterLabel)
                assert isinstance(storeField, StoreField)
                
                store.append(storeField)
                
        cls.dbPath = dbPath
        

    @classmethod
    def tearDownClass(cls):
        super(Test, cls).tearDownClass()
        if os.path.exists(cls.dbPath):
            os.remove(cls.dbPath)


    def test0001(self):
        
        store = Store(self.dbPath)
        agentFactory = AgentFactory()
        environmentFactory = EnvironmentFactory()
        buildParameterFactory = BuildParameterFactory()
        
        loader = Loader(agentFactory, buildParameterFactory, environmentFactory, store)
        assert isinstance(loader, Loader)
        
        for agent, buildParameter, epoch in loader.load("test%", None):
            assert isinstance(agent, Agent)
            assert isinstance(buildParameter, BuildParameter)
            

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test0001']
    unittest.main()