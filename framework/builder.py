'''
Created on 2020/07/09

@author: ukai
'''
from builtins import isinstance

from agent import Agent
from agent_factory import AgentFactory
from build_parameter import BuildParameter
from environment_factory import EnvironmentFactory
from trainer_factory import TrainerFactory
from store import Store
from store_field import StoreField


class Builder(object):
    '''
    classdocs
    '''


    def __init__(self, trainerFactory, agentFactory, environmentFactory, store, logger):
        '''
        Constructor
        '''
        
        assert isinstance(trainerFactory, TrainerFactory)
        assert isinstance(agentFactory, AgentFactory)
        assert isinstance(environmentFactory, EnvironmentFactory)
        assert isinstance(store, Store)
        
        self.environmentFactory = environmentFactory
        self.agentFactory = agentFactory
        self.trainerFactory = trainerFactory
        self.store = store
        self.logger = logger
                

    # <<public>>        
    def build(self, buildParameter):
        isinstance(buildParameter, BuildParameter)

        agent = self.agentFactory.create(buildParameter)
        environment = self.environmentFactory.create(buildParameter)        
        trainer = self.trainerFactory.create(buildParameter, agent, environment)
        
        nEpoch = buildParameter.nEpoch
        nIntervalSave = buildParameter.nIntervalSave
        
        epoch = 0
        self.save(agent, buildParameter, epoch)
        self.logger.info(agent, buildParameter, environment, epoch, trainer)
            
        while True:
            if epoch >= nEpoch:  
                break
            else:
                self.save(agent, buildParameter, epoch)
                self.logger.info(agent, buildParameter, environment, epoch, trainer)
            
            nEpochLoc = min(nIntervalSave, nEpoch - epoch) 
            for _ in range(nEpochLoc):
                trainer.train()
            self.save(agent, buildParameter, epoch)
            self.logger.info(agent, buildParameter, environment, epoch, trainer)            
            epoch += nEpochLoc 

    
    # <<private>>
    def save(self, agent, buildParameter, epoch):
        
        assert isinstance(agent, Agent)
        assert isinstance(buildParameter, BuildParameter)
        
        agentMemento = agent.createMemento()
        buildParameterMemento = buildParameter.createMemento()
        buildParameterLabel = buildParameter.label
                        
        storeField = StoreField(agentMemento, epoch, buildParameterMemento, buildParameterLabel)
        self.store.append(storeField)