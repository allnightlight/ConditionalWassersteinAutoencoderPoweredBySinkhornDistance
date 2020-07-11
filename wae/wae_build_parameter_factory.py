'''
Created on 2020/07/11

@author: ukai
'''
from sl_build_parameter_factory import SlBuildParameterFactory
from wae_build_parameter import WaeBuildParameter

class WaeBuildParameterFactory(SlBuildParameterFactory):
    '''
    classdocs
    '''


    def create(self):
        return WaeBuildParameter()