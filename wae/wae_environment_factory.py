'''
Created on 2020/07/11

@author: ukai
'''
from sl_environment_factory import SlEnvironmentFactory
from builtins import isinstance
from wae_build_parameter import WaeBuildParameter
from wae_environment import WaeEnvironment

class WaeEnvironmentFactory(SlEnvironmentFactory):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
        super(WaeEnvironmentFactory, self).__init__()
        
        
    def create(self, buildParameter):
        assert isinstance(buildParameter, WaeBuildParameter)
        
        environment = WaeEnvironment(nBatch = buildParameter.nBatch)
        environment.loadData()
        
        return environment