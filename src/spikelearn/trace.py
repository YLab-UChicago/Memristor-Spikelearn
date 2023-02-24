import numpy as np

class Trace:

    def __init__(self, N, t0, t1, tracelim):
        self.N = N
        self.t0 = t0
        self.t1 = t1
        self.tracelim = tracelim
        self.reset()

    def reset(self):
        self.t = np.zeros(self.N)

    def update(self, x):
        self.t*=self.t1
        self.t+=self.t0*x
        self.t[self.t>self.tracelim] = self.tracelim

    def __call__(self):
        return self.t


class BinaryTrace(Trace):
    def __init__(self, N, off, on):
        super().__init__(N, None, None, None)
        self.off=off
        self.on=on
        self.reset()

    def reset(self):
        self.t = np.ones(self.N)*self.off
    
    def update(self, x):
        self.t[x>0]=self.on

class IfRecentTrace(Trace):
    def __init__(self, N, off, on, duration):
        self.off=off
        self.on=on
        self.duration=duration
        super().__init__(N, None, None, None)

    def reset(self):
        self.t = np.ones(self.N)*self.duration
    
    def update(self, x):
        self.t+=1
        self.t[x>0]=0
    
    def call(self):
        return np.where(self.t<self.duration,self.on,self.off)

class ConstantTrace(Trace):
    def __init__(self, N, val, dtype=np.float64):
        self.val=val
        self.dtype=dtype
        super().__init__(N, None, None, None)
    
    def reset(self):
        self.t=np.ones(self.N, self.dtype)*self.val
    
    def update(self, x):
        pass

class ManualTrace(ConstantTrace):
    def __init__(self, N, val, dtype):
        super().__init__(N, val, dtype)

    def set_val(self, val):
        self.t[:]=val

class AccumulateTrace(Trace):
    def __init__(self, N):
        self.N=N

    def update(self, x):
        self.t+=x
