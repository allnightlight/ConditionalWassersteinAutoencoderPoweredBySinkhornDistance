'''
Created on 2020/06/30

@author: ukai
'''
from datetime import datetime
import sqlite3
import traceback

import torch

from conc_batch_data_environment import ConcBatchDataEnvironment
import numpy as np
from wae_environment import WaeEnvironment
from data_singleton import DataSingleton


class ConcEnvironment(WaeEnvironment):
    '''
    classdocs
    '''


    def __init__(self, dbFilePath, ev_tag, pv_tags, period_train_str, period_test_str, samplingIntervalMinute, nBatch):
        '''
        Constructor
            dbFilePath: str
            ev_tag: str
            pv_tags: list of strings
            period_train|test: tuple with two elements as string which must follow the format "%Y/%m/%d %H:%M"
        '''
        
        super(ConcEnvironment, self).__init__(nBatch)
        
        for elm in period_train_str + period_test_str:
            assert isinstance(elm, str), """\
            All the items of period_train and period_test must be defined as string and
            they have to follow the format "%Y/%m/%d %H:%M".\    
            """
        
        period_train = [*map(lambda xx: datetime.strptime(xx, "%Y/%m/%d %H:%M"), period_train_str)]
        period_test = [*map(lambda xx: datetime.strptime(xx, "%Y/%m/%d %H:%M"), period_test_str)]
        
        self.dbFilePath = dbFilePath
        self.ev_tag = ev_tag
        self.pv_tags = pv_tags
        self.period_train = period_train
        self.period_test = period_test
        self.period = (min(period_train[0], period_test[0])
                       , max(period_train[1], period_test[1]))
        self.timestamp = None
        self.dataX = None
        self.dataZ = None # EV[i,:] = (0,1) means the event occurrence between t[i] and t[i+1]
        self.samplingIntervalMinute = samplingIntervalMinute
        
        self.nX = len(pv_tags)
        self.nZ = 2
        
    
    # <<public>>            
    def loadData(self):
        
        tags = [*self.pv_tags, self.ev_tag]
        period = self.period
        
        dataInst = DataSingleton.getInstance(self.dbFilePath, period, tags, self.samplingIntervalMinute)

        data = dataInst.data
        timestamp = dataInst.timestamp
                        
        nSample = len(timestamp)
        
        dataPvRaw = data[:,:-1] # (*, nPv)        
        self.dataX = self.passDataPvToPreprocess(dataPvRaw) # (*, nPv')
        
        Zint = data[:,-1] # (*,), in {0,1,nan}        
        Zoh = np.zeros((nSample,2)) # (nSample, 2)        
        idxNaN = np.where(np.isnan(Zint))[0] # (*,)
        idxValid = np.where(~np.isnan(Zint))[0] # (*,)
        Zoh[idxValid[Zint[idxValid] == 1],1] = 1
        Zoh[idxValid[Zint[idxValid] == 0],0] = 1
        Zoh[idxNaN,:] = np.nan        
        self.dataZ = Zoh # (*, nZ)
        self.timestamp = timestamp
                
        return;
    
    # <<protected>>
    def generateIdxIterator(self):
        
        idx = np.where((self.timestamp >= self.period_train[0]) & (self.timestamp < self.period_train[1]))[0]
        idx = idx[~np.any(np.isnan(self.dataX[idx,:]), axis=-1)]
        idx = idx[~np.any(np.isnan(self.dataZ[idx,:]), axis=-1)] # not include nan for Pv and Event
        
        idxOff = idx[self.dataZ[idx,0] == 1] # index without event
        idxOn = idx[self.dataZ[idx,1] == 1] # index with event
                
        nSampleTrainOn = len(idxOn)
        for _ in range(nSampleTrainOn//self.nBatch):
            idxOnBatch = np.random.choice(idxOn, size=(self.nBatch,)) # (nBatch,)
            idxOffBatch = np.random.choice(idxOff, size=(self.nBatch,)) # (nBatch,)
            idxBatch = np.concatenate((idxOnBatch, idxOffBatch), axis=0) # (2*nBatch,)        
            yield idxBatch
        
    # <<private>>
    def passDataPvToPreprocess(self, dataPv):
        dataPvPreprocessed = dataPv
        return dataPvPreprocessed
    
    # <<public>>
    def getDataTrain(self):
        
        idx = np.where((self.timestamp >= self.period_train[0]) & (self.timestamp < self.period_train[1]))[0]
        idx = idx[~np.any(np.isnan(self.dataX[idx,:]), axis=-1)]
        idx = idx[~np.any(np.isnan(self.dataZ[idx,:]), axis=-1)]
                
        X = self.dataX[idx,:].astype(np.float32) # (*, nX)
        Z = self.dataZ[idx,:].astype(np.float32) # (*, nZ)
        timestamp = self.timestamp[idx]

        _X = torch.from_numpy(X) # (*, nX)
        _Z = torch.from_numpy(Z) # (*, nZ)
        
        return ConcBatchDataEnvironment(_X, _Z, timestamp)
    
    # <<public>>    
    def getDataTest(self):
        
        idx = np.where((self.timestamp >= self.period_test[0]) & (self.timestamp < self.period_test[1]))[0]
        idx = idx[~np.any(np.isnan(self.dataX[idx,:]), axis=-1)]
        idx = idx[~np.any(np.isnan(self.dataZ[idx,:]), axis=-1)]

        
        X = self.dataX[idx,:].astype(np.float32) # (*, nX)
        Z = self.dataZ[idx,:].astype(np.float32) # (*, nZ)
        timestamp = self.timestamp[idx]

        _X = torch.from_numpy(X) # (*, nX)
        _Z = torch.from_numpy(Z) # (*, nZ)
        
        return ConcBatchDataEnvironment(_X, _Z, timestamp)
