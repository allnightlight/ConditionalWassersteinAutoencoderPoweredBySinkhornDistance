'''
Created on 2020/08/02

@author: ukai
'''
from builtins import isinstance

from conc_build_parameter import ConcBuildParameter
from conc_environment_cs01a import ConcEnvironmentCs01a
from conc_environment_cs02a import ConcEnvironmentCs02a
from wae_environment_factory import WaeEnvironmentFactory


class ConcEnvironmentFactory(WaeEnvironmentFactory):
    '''
    classdocs
    '''

    def create(self, buildParameter):
        assert isinstance(buildParameter, ConcBuildParameter)
        
        if buildParameter.target_casestudy == "cs01a":
            environment = ConcEnvironmentCs01a(buildParameter.nBatch)

        if buildParameter.target_casestudy == "cs02a":
            environment = ConcEnvironmentCs02a(buildParameter.nBatch)
        
        environment.loadData() 
        return environment