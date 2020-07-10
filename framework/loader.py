'''
Created on 2020/07/10

@author: ukai
'''
from agent_factory import AgentFactory
from store import Store


class Loader(object):
    '''
    classdocs
    '''


    def __init__(self, agentFactory, buildParameterFactory, store):
        '''
        Constructor
        '''
        
        assert isinstance(agentFactory, AgentFactory)
        assert isinstance(store, Store)
        
        self.agentFactory = agentFactory
        self.buildParameterFactory = buildParameterFactory
        self.store = store
        
    def load(self, buildParameterLabel, epoch = None):
        
        for storeField in self.store.restore(buildParameterLabel, epoch):
            buildParameter = self.buildParameterFactory.create()            
            buildParameter.loadMemento(storeField.buildParameterMemento)
            
            agent = self.agentFactory.create(buildParameter)
            agent.loadMemento(storeField.agentMemento)
            
            yield agent, buildParameter, epoch        