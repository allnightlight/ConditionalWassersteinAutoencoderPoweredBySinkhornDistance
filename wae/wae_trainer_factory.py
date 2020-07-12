'''
Created on 2020/07/11

@author: ukai
'''
from sl_trainer_factory import SlTrainerFactory
from wae_agent import WaeAgent
from wae_build_parameter import WaeBuildParameter
from wae_environment import WaeEnvironment
from wae_trainer import WaeTrainer


class WaeTrainerFactory(SlTrainerFactory):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(WaeTrainerFactory, self).__init__()
        
        
    def create(self, buildParameter, agent, environment):
        
        assert isinstance(buildParameter, WaeBuildParameter)
        assert isinstance(environment, WaeEnvironment)
        assert isinstance(agent, WaeAgent)
        
        return WaeTrainer(agent, environment, reg_param = buildParameter.reg_param)