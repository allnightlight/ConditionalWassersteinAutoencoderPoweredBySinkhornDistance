'''
Created on 2020/07/11

@author: ukai
'''
import torch
from torch.optim import Adam
from sl_trainer import SlTrainer
from wae_agent import WaeAgent
from builtins import isinstance
from wae_environment import WaeEnvironment
from batch_data_agent import BatchDataAgent
from wae_batch_data_environment import WaeBatchDataEnvironment
from wae_batch_data_agent import WaeBatchDataAgent

class WaeTrainer(SlTrainer):
    '''
    classdocs
    '''


    def __init__(self, agent, environment):
        '''
        Constructor
        '''
        
        super(WaeTrainer, self).__init__(agent, environment)
        
        assert isinstance(agent, WaeAgent)
        assert isinstance(environment, WaeEnvironment)
        
        self.optimizer = Adam(agent.parameters())
        
        
    def update(self, batchDataEnvironment, batchDataAgent):
        
        assert isinstance(batchDataEnvironment, WaeBatchDataEnvironment)        
        assert isinstance(batchDataAgent, WaeBatchDataAgent)
        
        _XHat = batchDataAgent._XHat # (*, nX)
        _X = batchDataEnvironment._X # (*, nX)
        
        _loss = torch.mean((_X-_XHat)**2) # (,)
        
        self.optimizer.zero_grad()
        _loss.backward()
        self.optimizer.step()
        