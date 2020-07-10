'''
Created on 2020/07/10

@author: ukai
'''

from agent import Agent
from batch_data_out import BatchDataOut
from batch_data_in import BatchDataIn


class SlAgent(Agent):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(SlAgent, self).__init__()

    # <<abstract>>
    def forward(self, batchDataIn):
        assert isinstance(batchDataIn, BatchDataIn)
        batchDataOut = BatchDataOut()
        return batchDataOut
    