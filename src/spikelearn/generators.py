import numpy as np


class Generator:
    """Base class for spike generators
    
    """
    def __init__(self, N, r0, scale=1.0):
        """Instantiates a spike generator

        Args:
            N : number of parallel streams
            r0: array with input rates, with values between 0 and 1
            scale: scalar factor to control the degree of sparsity

        """
        self.N = N
        self.r0 = r0
        self.scale = scale
        self.init()

    def __call__(self, r0=None):
        """Returns an array of spikes

        Args:
            r0 : optional, array with input rates

        Returns:
            An array of spikes (1 if spiking 0 otherwise)
        """
        if r0 is not None:
            self.p0 = r0
        return self._next_spike()

    def _next_spike(self):
        raise NotImplementedError()

    def init(self):
        pass

class Poisson(Generator):
    """Poisson spike generator

    Generates an array of input spikes from a rate input value

    Attributes:
        N : number of spike trains
        r0 : array of rates used to generate the spike trains
        scale : scalar factor used to control the degree of sparsity
    
    """
    
    def _next_spike(self):
        s = np.random.random(self.N)
        return np.where(s > self.scale*self.r0, 0, 1)


class Periodic(Generator):
    """Periodic spike generator

    Generates an array of deterministic periodic spike trains

    Attributes:
        N : number of spike trains
        r0 : array of rates used to generate the spike trains
        scale : scalar factor used to control the degree of sparsity

    """

    def init(self):
        self.val = np.zeros(self.N)

    def _next_spike(self):
        self.val += self.scale*self.r0
        out = np.where(self.val>1, 1, 0)
        self.val[self.val>1] = 0
        return out


class SpikeSequence(Generator):
    """Spike sequence generator
    
    Generates an array of spike sequences. Sequences repeat
    over time. 

    Attributes:
        N : number of spike trains
        r0 : 2D array with the spike sequences
    """

    def __init__(self, N, r0):
        """Instantiates a spike sequence generator

        Args:
            N : number of parallel streams
            r0: 2D array with spike sequences

        """
        super().__init__(N, r0)

    def init(self):
        self.time_step = 0
        self.pulse_length = self.r0.shape[1]

    def _next_spike(self):
        out = self.r0[:,self.time_step]
        self.time_step += 1
        if self.time_step == self.pulse_length:
            self.time_step = 0
        return out
