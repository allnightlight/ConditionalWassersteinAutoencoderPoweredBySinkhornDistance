'''
Created on 2020/07/11

@author: ukai
'''
from batch_data_agent import BatchDataAgent

class WaeBatchDataAgent(BatchDataAgent):
    '''
    classdocs
    '''


    def __init__(self, _Xi, _XiHat, _XHat):
        '''
        Constructor
        '''
        super(WaeBatchDataAgent, self).__init__()
        
        # _Xi, _XiHat: (*, nXi)
        # _XHat: (*, nX)
        
        assert len(_XHat.shape) == 2
        assert len(_Xi.shape) == 2        
        assert _Xi.shape == _XiHat.shape
        assert _XHat.shape[0] == _Xi.shape[0]
        
        self._Xi = _Xi          
        self._XiHat = _XiHat
        self._XHat = _XHat