'''
Created on 2020/07/11

@author: ukai
'''
import torch
from torch.optim import Adam
from sl_trainer import SlTrainer
from wae_agent import WaeAgent
from builtins import isinstance
from wae_environment import WaeEnvironment
from wae_batch_data_environment import WaeBatchDataEnvironment
from wae_batch_data_agent import WaeBatchDataAgent
from batch_data_agent import BatchDataAgent

class WaeTrainer(SlTrainer):
    '''
    classdocs
    '''


    def __init__(self, agent, environment, reg_param, tol_sinkhorn = 0.1, eps_given_sinkhorn = 0.1, max_itr_sinkhorn = 2**10):
        '''
        Constructor
        '''
        
        super(WaeTrainer, self).__init__(agent, environment)
        
        assert isinstance(agent, WaeAgent)
        assert isinstance(environment, WaeEnvironment)
        
        self.optimizer = Adam(agent.parameters())
        self.reg_param = reg_param
        self.tol_sinkhorn = tol_sinkhorn
        self.eps_given_sinkhorn = eps_given_sinkhorn
        self.max_itr_sinkhorn = max_itr_sinkhorn
        
        
    def update(self, batchDataEnvironment, batchDataAgent):
        
        assert isinstance(batchDataEnvironment, WaeBatchDataEnvironment)        
        assert isinstance(batchDataAgent, WaeBatchDataAgent)

        nZ = self.environment.nZ
        
        _XHat = batchDataAgent._XHat # (*, nX)
        _X = batchDataEnvironment._X # (*, nX)
        
        _Z = batchDataEnvironment._Z # (*, nZ)
        _XiHat = batchDataAgent._XiHat # (*, nXi)
        _Xi = batchDataAgent._Xi # (*, nXi)
        
        nBatch = _Z.shape[0]
        
        dist = [None,] * nZ
        for k1 in range(nZ):
            _XiGivenZ = _Xi[_Z[:,k1]==1,:] # (*', nXi)
            _XiHatGivenZ = _XiHat[_Z[:,k1]==1,:] # (*', nXi)
            nBatchGivenZ = torch.sum(_Z[:,k1]==1)
            if nBatchGivenZ > 0:
                dist[k1], _ = self.measure_distance(_XiGivenZ, _XiHatGivenZ)
                dist[k1] *= nBatchGivenZ
            else:
                dist[k1] = torch.zeros()
        
        _dist = torch.sum(torch.stack(dist, axis=0))/nBatch
        
        _loss = torch.mean((_X-_XHat)**2) + _dist * self.reg_param # (,)
        
        self.optimizer.zero_grad()
        _loss.backward()
        self.optimizer.step()
        
    def measure_distance(self, _X0, _X1):
        # _X0: (*, Nx), _X1: (*, Nx)
        
        def robust_sinkhorn_iteration(_M, _p, _q, _eps, tol, max_itr):
            _alpha = _p * 0
            _beta = _q * 0
            cnt = 0
        
            while True:
    
                _P = torch.exp(-(_M-_alpha-_beta)/_eps -1)
                _qhat = torch.sum(_P, dim=0, keepdim=True)
                _err = torch.sum(torch.abs(_qhat - _q))
    
                if _err < tol or cnt >= max_itr:
                    break
                else:
                    cnt += 1
    
                _delta_row = torch.min(_M - _alpha, dim=0, keepdim = True)[0]
                _beta = _eps + _eps * torch.log(_q) + _delta_row             - _eps * torch.log( torch.sum( torch.exp(-(_M-_alpha-_delta_row)/_eps ), dim=0, keepdim = True ) )
                _delta_col = torch.min(_M - _beta, dim=1, keepdim = True)[0]
                _alpha = _eps + _eps * torch.log(_p) + _delta_col             - _eps * torch.log( torch.sum( torch.exp( -(_M - _beta - _delta_col)/_eps  ), dim=1, keepdim = True )  )
    
            if cnt == max_itr:
                #print('Warning: Sinkhorn iteration did not converge within the given max iteration number: %d.' % max_itr)
                pass
            _dist = torch.sum(_p * _alpha) + torch.sum(_q * _beta) - _eps
            return _dist, cnt
        
        tol = self.tol_sinkhorn
        eps_given = self.eps_given_sinkhorn
        max_itr = self.max_itr_sinkhorn
        
        _M01 = torch.sum(torch.abs(_X0.unsqueeze(1) - _X1), dim=2)
        _M00 = torch.sum(torch.abs(_X0.unsqueeze(1) - _X0), dim=2)
        _M11 = torch.sum(torch.abs(_X1.unsqueeze(1) - _X1), dim=2)
    
        _p = 1/_X0.shape[0] * torch.ones(_X0.shape[0])
        _q = 1/_X1.shape[0] * torch.ones(_X1.shape[0])
    
        _eps = torch.tensor(eps_given)
            
        _dist00, cnt_00 = robust_sinkhorn_iteration(_M00, _p.unsqueeze(1), _p.unsqueeze(0), _eps, tol, max_itr=max_itr)
        _dist11, cnt_11 = robust_sinkhorn_iteration(_M11, _q.unsqueeze(1), _q.unsqueeze(0), _eps, tol, max_itr=max_itr)
        _dist01, cnt_01 = robust_sinkhorn_iteration(_M01, _p.unsqueeze(1), _q.unsqueeze(0), _eps, tol, max_itr=max_itr)
        
        _wdist = 2. * _dist01 - _dist00 - _dist11
        
        return _wdist, (cnt_00, cnt_11, cnt_01)