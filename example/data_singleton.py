'''
Created on 2020/07/13

@author: ukai
'''

from builtins import classmethod
from datetime import datetime
import json
import sqlite3
import traceback

import numpy as np


class DataSingleton(object):
    '''
    classdocs
    '''
    
    _instances = {}

    # << private>>
    def __init__(self, dbFilePath, period, tags, samplingIntervalMinute):
        '''
        Constructor
            period is defined as list, of which items are defined as !!datetime!!
        '''
        
        print("""\
        
======================================
A dataset would load from the DB in {0} 
with the following configuration:
period: {1} - {2}
tags: {3}
samplingIntervalMinute: {4}
======================================        
                
""".format(
    dbFilePath
    , datetime.strftime(period[0], "%Y/%m/%d %H:%M")
    , datetime.strftime(period[1], "%Y/%m/%d %H:%M")
    , " ".join(tags)
    , samplingIntervalMinute
    ))
        
        sql = """
Select 
    d.value
    , timestamp
    From DataTable d
        Where d.tag in ({0})
        And timestamp >= ?
        And timestamp < ?
        And Cast(strftime('%M', timestamp) as int) % {1} == 0
    Order by d.timestamp_id, d.tag
""".format(",".join(map(lambda xx: '"%s"' % xx, tags)), samplingIntervalMinute)

        conn = None
        data = None
        try:
            conn = sqlite3.connect(dbFilePath, detect_types = sqlite3.PARSE_COLNAMES|sqlite3.PARSE_DECLTYPES)
            cur = conn.cursor()
            cur.execute(sql, period)
            data, timestamp = zip(*cur.fetchall())        
            data = np.array(data, dtype=np.float32).reshape(-1, len(tags)) # (nSample, nTag)
            timestamp = np.array([*map(lambda xx: datetime.strptime(xx, "%Y-%m-%d %H:%M:%S"),
                                       timestamp)]).reshape(-1, len(tags)) # (nSample, nTag)
            timestamp = timestamp[:,0] # (nSample,)
        except:
            traceback.print_exc()
            data = None
        finally:
            if conn is not None:                
                conn.close()

        assert data is not None, "FAILED TO LOADING DATA FROM THE GIVEN DB: %s" % self.dbFilePath
                        
        self.data = data[:, np.argsort(np.argsort(tags))]
        self.timestamp = timestamp


    @classmethod
    def getInstance(cls, dbFilePath, period, tags, samplingIntervalMinute):
        
        key = json.dumps(dict(            
            dbFilePath = dbFilePath 
            , t0 = str(period[0])
            , t1 = str(period[1]) 
            , tags = tags
            , samplingIntervalMinute = samplingIntervalMinute            
            ))
                
        if key in cls._instances: 
            inst = cls._instances[key]
        else:
            inst = DataSingleton(dbFilePath, period, tags, samplingIntervalMinute)
            cls._instances[key] = inst 
        
        return inst