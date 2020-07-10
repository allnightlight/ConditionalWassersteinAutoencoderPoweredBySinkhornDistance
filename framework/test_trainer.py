'''
Created on 2020/07/09

@author: ukai
'''
import unittest

from trainer import Trainer
from agent import Agent
from environment import Environment


class Test(unittest.TestCase):


    def setUp(self):
        unittest.TestCase.setUp(self)
        
        self.trainers = []
        self.trainers.append(Trainer())

    def test001(self):
        
        for trainer in self.trainers:
            assert isinstance(trainer, Trainer)
            
            agent = Agent()
            environment = Environment()

            try:            
                trainer.train(agent, environment)
            except NotImplementedError as ex:
                pass
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()