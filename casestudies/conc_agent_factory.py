'''
Created on 2020/08/02

@author: ukai
'''
from conc_agent_cs01a import ConcAgentCs01a
from conc_agent_cs02a import ConcAgentCs02a
from conc_agent_cs03a import ConcAgentCs03a
from conc_build_parameter import ConcBuildParameter
from conc_environment_cs01a import ConcEnvironmentCs01a
from conc_environment_cs02a import ConcEnvironmentCs02a
from conc_environment_cs03a import ConcEnvironmentCs03a
from wae_agent_factory import WaeAgentFactory


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
                            , cluster_interval = buildParameter.cluster_interval
                            , activation = buildParameter.activation)
            
        if buildParameter.target_casestudy == "cs02a":
            
            assert isinstance(environment, ConcEnvironmentCs02a)
        
            return ConcAgentCs02a(nX = environment.nX
                            , nZ = environment.nZ
                            , nH = buildParameter.nH
                            , nXi = buildParameter.nXi
                            , nLayer = buildParameter.nLayer
                            , cluster_interval = buildParameter.cluster_interval
                            , activation = buildParameter.activation)
            
        if buildParameter.target_casestudy == "cs03a":
            
            assert isinstance(environment, ConcEnvironmentCs03a)
        
            return ConcAgentCs03a(nX = environment.nX
                            , nZ = environment.nZ
                            , nH = buildParameter.nH
                            , nXi = buildParameter.nXi
                            , nLayer = buildParameter.nLayer
                            , cluster_interval = buildParameter.cluster_interval
                            , activation = buildParameter.activation)