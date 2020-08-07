'''
Created on 2020/08/02

@author: ukai
'''
from conc_build_parameter import ConcBuildParameter
from wae_agent_factory import WaeAgentFactory
from conc_environment_cs01a import ConcEnvironmentCs01a
from conc_agent_cs01a import ConcAgentCs01a


class ConcAgentFactory(WaeAgentFactory):
    '''
    classdocs
    '''


    def create(self, buildParameter, environment):
        
        assert isinstance(buildParameter, ConcBuildParameter)
        
        if buildParameter.target_casestudy == "cs01a":
            
            assert isinstance(environment, ConcEnvironmentCs01a)
        
            return ConcAgentCs01a(nX = environment.nX
                            , nZ = environment.nZ
                            , nH = buildParameter.nH
                            , nXi = buildParameter.nXi
                            , nLayer = buildParameter.nLayer
                            , cluster_interval = buildParameter.cluster_interval)