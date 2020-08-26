'''
Created on 2020/07/11

@author: ukai
'''
from sl_agent_factory import SlAgentFactory
from wae_build_parameter import WaeBuildParameter
from builtins import isinstance
from wae_environment import WaeEnvironment
from wae_agent import WaeAgent


class WaeAgentFactory(SlAgentFactory):
    '''
    classdocs
    '''


    def __init__(self):
        super(WaeAgentFactory, self).__init__()
        
        
    def create(self, buildParameter, environment):
        
        assert isinstance(buildParameter, WaeBuildParameter)
        assert isinstance(environment, WaeEnvironment)
        
        return WaeAgent(nX = environment.nX
                        , nZ = environment.nZ
                        , nH = buildParameter.nH
                        , nXi = buildParameter.nXi
                        , nLayer = buildParameter.nLayer
                        , cluster_interval = buildParameter.cluster_interval
                        , activation = buildParameter.activation)