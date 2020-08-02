'''
Created on 2020/08/02

@author: ukai
'''
from wae_environment_factory import WaeEnvironmentFactory
from builtins import isinstance
from conc_build_parameter import ConcBuildParameter
from conc_environment_cs01a import ConcEnvironmentCs01a

class ConcEnvironmentFactory(WaeEnvironmentFactory):
    '''
    classdocs
    '''

    def create(self, buildParameter):
        assert isinstance(buildParameter, ConcBuildParameter)
        
        if buildParameter.target_casestudy == "cs01a":
            environment = ConcEnvironmentCs01a(buildParameter.nBatch)
        
        environment.loadData() 
        return environment