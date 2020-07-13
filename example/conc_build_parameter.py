'''
Created on 2020/07/13

@author: ukai
'''
from wae_build_parameter import WaeBuildParameter

class ConcBuildParameter(WaeBuildParameter):
    '''
    classdocs
    '''


    def __init__(self, nIntervalSave=2 ** 4, nEpoch=2 ** 7, label="None", nBatch=2 ** 5, nH=2 ** 3, nXi=2, reg_param=0.1, cluster_interval=3.0,
                 dbFilePath = None
                 ,ev_tag = None
                 ,pv_tags = None
                 ,period_train_str = None
                 ,period_test_str = None
                 ,samplingIntervalMinute = None
                 ):
        WaeBuildParameter.__init__(self, nIntervalSave=nIntervalSave, nEpoch=nEpoch, label=label, nBatch=nBatch, nH=nH, nXi=nXi, reg_param=reg_param, cluster_interval=cluster_interval)
        
        self.dbFilePath=dbFilePath
        self.ev_tag=ev_tag
        self.pv_tags=pv_tags
        self.period_train_str=period_train_str
        self.period_test_str=period_test_str
        self.samplingIntervalMinute=samplingIntervalMinute
