'''
Created on 2020/07/09

@author: ukai
'''
from agent import Agent
from environment import Environment


class Trainer(object):
    '''
    classdocs
    '''
    
    # <<public>>
    def train(self, agent, environment):
        assert isinstance(agent, Agent)
        assert isinstance(environment, Environment)