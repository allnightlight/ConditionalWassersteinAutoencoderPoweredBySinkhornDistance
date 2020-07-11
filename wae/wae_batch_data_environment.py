'''
Created on 2020/07/11

@author: ukai
'''
from batch_data_environment import BatchDataEnvironment

class WaeBatchDataEnvironment(BatchDataEnvironment):
    '''
    classdocs
    '''


    def __init__(self, _X, _Z):
        '''
        Constructor
        '''
        
        # _X: (*, nX)
        # _Z: (*, nZ)
        
        assert len(_X.shape) == 2
        assert len(_Z.shape) == 2
        assert _X.shape[0] == _Z.shape[0]
        
        self._X = _X
        self._Z = _Z