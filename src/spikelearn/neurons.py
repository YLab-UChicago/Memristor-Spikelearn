import numpy as np

def hard(x):
    return np.where(x>=0, 1.0, 0.0).astype(x.dtype)


class SpikingLayer:
    """Implement a leaky integrate and fire"""

    def __init__(self, N, tau, v0=1):
        self.N = N
        self.tau = tau
        self.a = np.exp(-1./tau)
        self.b = 1-self.a
        self.v0 = v0*np.ones(N)
        self.v = np.zeros(N)
        self.s = np.zeros(N)

    def __call__(self, *x):
        xtot = sum(x)
        self.v = (1-self.s) * (self.a * self.v + self.b * xtot)
        self.s = hard(self.v-self.v0)
        return self.s

    def step(self, *x):
        return self(x)

    def reset(self):
        self.v = np.zeros(self.N)
        self.s = np.zeros(self.N)

class IntFireLayer:
    def __init__(self, N, v0=1):
        self.N=N
        self.v0=v0*np.ones(N)
        self.v=np.zeros(N)
        self.s=np.zeros(N,dtype=np.int32)
    
    def __call__(self, *x):
        xtot=sum(x)
        self.v=(self.v+xtot)
        self.s[:]=hard(self.v-self.v0)
        self.v=np.where(self.s>0.5, 0.0, self.v)
        return self.s

    def step(self, *x):
        return self(x)

    def reset(self):
        self.v = np.zeros(self.N)
        self.s = np.zeros(self.N,dtype=np.int32)

class LatchLayer:
    def __init__(self, N):
        self.N=N
        self.v=np.zeros(N)
        self.s=np.zeros(N)
    
    def __call__(self, *x):
        xtot = sum(x)
        self.v[xtot>0]=1
        self.s[:]=self.v
        return self.s
    
    def step(self, *x):
        return self(x)
    
    def reset(self):
        self.v[:]=0
        self.s[:]=0

class LeakyIntegralSoftmax:
    def __init__(self, Nloc, Nch, tau, threshinit, threshstep, global_skip, local_skip):
        self.Nloc=Nloc
        self.Nch=Nch
        self.threshinit=threshinit
        #self.thresh=threshinit
        self.thresh=np.empty((Nloc,))
        self.thresh[:]=threshinit
        self.threshstep=threshstep
        self.s=np.zeros(Nloc*Nch)
        self.stepsinceagg=0
        self.ifactive=np.zeros(Nloc, dtype=np.int8)
        self.global_skip=global_skip
        self.local_skip=local_skip
        self.exp1tau=np.exp(-1/tau)
        self.x_mem=np.zeros((Nloc, Nch))
    def reset(self):
        #self.thresh=self.threshinit
        self.thresh[:]=self.threshinit
        self.ifactive[:]=0
        self.stepsinceagg=0
    def aggregate(self):
        if self.global_skip:
            self.thresh[:]=self.thresh.mean()+self.stepsinceagg*(self.ifactive==0).sum()*self.threshstep/self.Nloc
        else:
            self.thresh[:]=self.thresh.mean()
        self.ifactive[:]=0
        self.stepsinceagg=0
        self.x_mem[:]=0
    def __call__(self, *x):
        xtot = np.reshape(sum(x), (self.Nloc, self.Nch))
        self.ifactive[xtot.sum(axis=1)!=0]=1
        self.stepsinceagg+=1
        
        self.x_mem*=self.exp1tau
        self.x_mem+=xtot
        xtot=self.x_mem

        # xtot-=xtot.mean(axis=1)[:,None] # doesn't affect result of softmax
        xtot=np.exp(xtot-xtot.max(axis=1)[:,None]) # slow
        # xtot=xtot/np.sum(xtot, axis=1)[:,None]
        spikes=xtot>(self.thresh*np.sum(xtot, axis=1))[:,None]
        self.s[:]=np.reshape(spikes,[-1])
        if self.local_skip:
            self.thresh+=(np.sum(spikes, axis=1)-(xtot.sum(axis=1)!=0))*self.threshstep
        else:
            self.thresh+=(np.sum(spikes, axis=1)-1)*self.threshstep
        return self.s

class NormalizedLIF(SpikingLayer):
    def __init__(self, N, num_shape, den_shape, tau, v0=1):
        super().__init__(N, tau, v0)
        self.b=1
        self.num_shape=num_shape
        self.den_shape=den_shape

    def __call__(self, *x):
        num,den = x
        num=num.reshape(self.num_shape)
        den=den.reshape(self.den_shape)
        self.v = (1-self.s) * self.a * self.v + (num/np.sqrt(np.where(den==0, 1, den))).flatten()
        self.s = hard(self.v-self.v0)
        return self.s

class IdentityLayer:
    def __init__(self, N):
        self.N=N
        self.s=np.zeros(N)
    def reset(self):
        self.s=np.zeros(self.N)
    def __call__(self, *x):
        xtot = sum(x)
        self.s[:]=xtot
        return self.s

class SecondOrderLayer:
    """
    A second-order neuron. See https://doi.org/10.3389/fncom.2015.00099 and their demo code
    Imaginary part of inputs are used as inhibitory
    """
    def __init__(self, N, tau, vth, vreset, vrest, v_init, t_refrac, tau_e, tau_i, v_offset_i):
        dtype=np.float32
        self.dtype=dtype
        timetype=np.float32
        self.timetype=timetype
        self.N=N
        self.tau=dtype(tau)
        self.vth=dtype(vth)
        self.vreset=dtype(vreset)
        self.vrest=dtype(vrest)
        self.v_init=dtype(v_init)
        self.t_refrac=timetype(t_refrac)
        self.tau_e=dtype(tau_e)
        self.tau_i=dtype(tau_i)
        self.v_offset_i=dtype(v_offset_i)
        self.v=np.full(N, v_init, dtype=dtype)
        self.s=np.zeros(N, dtype=dtype)
        self.refrac_remain=np.zeros(N, dtype=timetype)
        self.ge=np.zeros(N, dtype=dtype)
        self.gi=np.zeros(N, dtype=dtype)
    
    def reset(self):
        self.v[:]=self.v_init
        self.s[:]=0.
        self.ge[:]=0.
        self.gi[:]=0.

    def __call__(self, *x):
        xtot = sum(x)
        self.ge+=xtot.real
        self.gi+=xtot.imag
        del xtot

        self.refrac_remain[self.s!=0]=self.t_refrac
        in_refrac=self.refrac_remain>0

        self.v-=((self.v-self.vrest)+self.v*self.ge+(self.v+self.v_offset_i)*self.gi)*(1-np.exp(-1/self.tau))
        self.v=np.maximum(self.v, self.vreset) # unsure: clamp or not
        self.ge*=np.exp(-1/self.tau_e)
        self.gi*=np.exp(-1/self.tau_i)
        
        self.s[:]=0
        self.s[self.v>=self.vth]=1
        self.v[in_refrac]=self.vreset
        self.refrac_remain[in_refrac]-=1
        self.s[in_refrac]=0
        return self.s

    def step(self, *x):
        return self(x)

class HomeostasisLayer(SecondOrderLayer):
    def __init__(self, theta_tau, theta_delta, theta_init, N, tau, vth, vreset, vrest, v_init, t_refrac, tau_e, tau_i, v_offset_i):
        super().__init__(N, tau, vth, vreset, vrest, v_init, t_refrac, tau_e, tau_i, v_offset_i)
        dtype=self.dtype
        self.theta_tau=dtype(theta_tau)
        self.theta_delta=dtype(theta_delta)
        self.theta_init=theta_init
        self.theta=np.full(self.N, theta_init, dtype=dtype)
        self.vth_base=self.vth
        self.vth=np.empty(self.N, dtype=dtype)
    
    def __call__(self, *x):
        self.vth[:]=self.theta+self.vth_base
        super().__call__(*x)
        self.theta*=np.exp(-1/self.theta_tau)
        self.theta+=self.s*self.theta_delta
        return self.s
    
    def reset(self):
        super().reset()
        self.theta[:]=self.theta_init


