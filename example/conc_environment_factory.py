'''
Created on 2020/07/13

@author: ukai
'''
from wae_environment_factory import WaeEnvironmentFactory
from conc_environment import ConcEnvironment

class ConcEnvironmentFactory(WaeEnvironmentFactory):
    '''
    classdocs
    '''


    def create(self, buildParameter):
        
        names="dbFilePath,ev_tag,pv_tags,period_train_str,period_test_str,samplingIntervalMinute,nBatch".split(",")
        arg = {name: buildParameter.__dict__[name] for name in names}
        
        environment = ConcEnvironment(**arg)
        environment.loadData()
        
        return environment
        