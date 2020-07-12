'''
Created on 2020/07/11

@author: ukai
'''
from sl_build_parameter import SlBuildParameter

class WaeBuildParameter(SlBuildParameter):
    '''
    classdocs
    '''


    def __init__(self, nIntervalSave=2 ** 4, nEpoch=2 ** 7, label="None", nBatch = 2**5, nH = 2**3, nXi = 2, reg_param = 0.1):
        SlBuildParameter.__init__(self, nIntervalSave=nIntervalSave, nEpoch=nEpoch, label=label)
        
        self.nBatch = nBatch
        self.nH = nH
        self.nXi = nXi
        self.reg_param = reg_param