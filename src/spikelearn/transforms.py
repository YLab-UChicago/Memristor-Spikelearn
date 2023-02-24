import numpy as np

class LowPass:
    """
    Implement a low pass function with optional delay

    """

    def __init__(self, N, tau, delay=False):
        """
        Parameters
        ----------

        N : int
            Number of neurons
        tau : float
            Characteristic time
        delay : bool
            If true, delay inputs for an additional timestep

        """
        self.beta  = np.exp(-1/tau)
        self.N = N
        self.value = np.zeros(self.N)
        self.delay = delay

    def reset(self):
        self.value = np.zeros(self.N)

    def __call__(self, x):
        if self.delay:
            toreturn = self.value.copy()
            self.value = (1-self.beta)*x + self.beta*self.value
            return toreturn
        else:
            self.value = (1-self.beta)*x + self.beta*self.value
            return self.value


class PassThrough:

    """
    Pass-through function with optional delay

    """

    def __init__(self, N, delay=False):
        self.N = N
        self.value = np.zeros(N)
        self.delay = delay

    def reset(self):
        self.value = np.zeros(self.N)

    def __call__(self, x):
        if self.delay:
            toreturn = self.value.copy()
            self.value = x
            return toreturn
        else:
            self.value = x
            return self.value

