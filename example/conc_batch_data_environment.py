'''
Created on 2020/07/13

@author: ukai
'''
from wae_batch_data_environment import WaeBatchDataEnvironment

class ConcBatchDataEnvironment(WaeBatchDataEnvironment):
    '''
    classdocs
    '''


    def __init__(self, _X, _Z, timestamp):
        '''
        Constructor
        '''
        super(ConcBatchDataEnvironment, self).__init__(_X, _Z)
        
        self.timestamp = timestamp
        