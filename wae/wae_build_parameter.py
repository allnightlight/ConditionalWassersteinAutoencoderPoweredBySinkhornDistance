'''
Created on 2020/07/11

@author: ukai
'''
from sl_build_parameter import SlBuildParameter

class WaeBuildParameter(SlBuildParameter):
    '''
    classdocs
    '''


    def __init__(self, nIntervalSave=2 ** 4, nEpoch=2 ** 7, label="None", nBatch = 2**5, nH = 2**3, nXi = 2, nLayer = 1, reg_param = 0.1, cluster_interval = 3.0, activation = "relu", tol_sinkhorn = 0.1, eps_given_sinkhorn = 0.1):
        SlBuildParameter.__init__(self, nIntervalSave=nIntervalSave, nEpoch=nEpoch, label=label)
        
        self.nBatch = nBatch
        self.nH = nH
        self.nXi = nXi
        self.nLayer = nLayer
        self.reg_param = reg_param
        self.cluster_interval = cluster_interval        
        self.activation = activation
        self.tol_sinkhorn = tol_sinkhorn
        self.eps_given_sinkhorn = eps_given_sinkhorn