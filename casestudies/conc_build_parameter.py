'''
Created on 2020/08/02

@author: ukai
'''
from wae_build_parameter import WaeBuildParameter

class ConcBuildParameter(WaeBuildParameter):
    
    def __init__(self, nIntervalSave=2 ** 4, nEpoch=2 ** 7, label="None", nBatch=2 ** 5, nH=2 ** 3, nXi=2, reg_param=0.1, cluster_interval=3.0, target_casestudy = "cs01a"):
        WaeBuildParameter.__init__(self, nIntervalSave=nIntervalSave, nEpoch=nEpoch, label=label, nBatch=nBatch, nH=nH, nXi=nXi, reg_param=reg_param, cluster_interval=cluster_interval)
        
        self.target_casestudy = target_casestudy
    