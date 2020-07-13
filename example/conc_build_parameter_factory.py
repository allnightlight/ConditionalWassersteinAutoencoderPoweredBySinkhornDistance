'''
Created on 2020/07/13

@author: ukai
'''
from wae_build_parameter_factory import WaeBuildParameterFactory
from conc_build_parameter import ConcBuildParameter

class ConcBuildParameterFactory(WaeBuildParameterFactory):
    '''
    classdocs
    '''


    def create(self):
        return ConcBuildParameter()