"""Persist exact submitted H40 labels and their prefix cursor; no replanning."""
import numpy as np


class SavedWindow:
    def __init__(self,action,index=0):
        self.action=np.asarray(action,dtype=np.float64).copy();self.index=index
        if self.action.shape!=(40,13) or not np.isfinite(self.action).all() or not 0<=index<=40:
            raise ValueError('FINITE_H40_WINDOW_AND_CURSOR')

    def take(self,count):
        if count<0 or self.index+count>40:raise ValueError('PREFIX_CURSOR_OUT_OF_BOUNDS')
        result=self.action[self.index:self.index+count].copy();self.index+=count;return result

    def snapshot(self):return dict(schema='SAVED_EEF13_WINDOW_V1',action=self.action.tolist(),index=self.index)

    @classmethod
    def restore(cls,value):
        if value['schema']!='SAVED_EEF13_WINDOW_V1':raise ValueError('WINDOW_SCHEMA')
        return cls(value['action'],value['index'])
