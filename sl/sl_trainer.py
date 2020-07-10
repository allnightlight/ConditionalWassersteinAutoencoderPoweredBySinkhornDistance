'''
Created on 2020/07/10

@author: ukai
'''
from builtins import isinstance

from sl_agent import SlAgent
from sl_environment import SlEnvironment
from trainer import Trainer
from batch_data_in import BatchDataIn
from batch_data_out import BatchDataOut


class SlTrainer(Trainer):
    '''
    classdocs
    '''

    def __init__(self, agent, environment):
        '''
        Constructor
        '''
        super(SlTrainer, self).__init__()
        
        assert isinstance(environment, SlEnvironment)
        assert isinstance(agent, SlAgent)
        
        self.agent = agent
        self.environment = environment
        
         
    def train(self):
        
        for batchDataIn in self.environment.generateBatchDataIterator():            
            batchDataOut = self.agent.forward(batchDataIn)
            self.update(batchDataIn, batchDataOut)
        
    # <<abstract>>    
    def update(self, batchDataIn, batchDataOut):
        assert isinstance(batchDataIn, BatchDataIn)
        assert isinstance(batchDataOut, BatchDataOut)
        return