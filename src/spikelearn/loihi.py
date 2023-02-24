from .synapses import BaseSynapse

class LoihiSynapse(BaseSynapse):
    """General plasticity rule inspired in that of Intel's Loihi chip

    Synaptic delays are ignored. Otherwise the rule follows the
    same code shown in Intel's IEEE Access Loihi paper.
    """

    def __init__(self, Ne, No, W0, tre, tro, wrule, transform=None,
        tagrule=None, Wlim=1, taglim=1,
        tre2=None, trm=None, tro2=None, tro3=None, tracelim=10):
        """
        Parameters
        ----------

        Ne : int
            number of presynaptic neurons
        No : int
            number of postsynaptic neurons
        W0 : np.array
            a 2D array with the initial synaptic weights
        tre : tuple
            presynaptic trace tuple
        tro : tuple 
            postsynaptic trace tuple
        wrule : list
            plasticity rule for synaptic weights
        lowpass : bool
            applies an alpha transform to inputs
        beta : float
            frequency of the alpha transform
        delay : bool
            if True, applies a 1 timestep delay
        tagrule : list
            plasticity rule for synaptic tag
        Wlim : float
            clamping parameter for synaptic weights
        taglim : float
            clamping parameter for synaptic tag
        tre2 :  tuple
            second presynaptic trace tuple 
        trm : tuple
            modulatory trace tuple
        tro2 : tuple
            second postsynaptic trace tuple
        tro3 : tuple
            third postsynaptic trace tuple
        tracelim : float
            clamping parameter for synaptic traces
        
        """

        super().__init__(Ne, No, W0, transform)

        self.tre = tre
        self.tro = tro
        self.tre2 = tre2
        self.trm = trm
        self.tro2 = tro2
        self.tro3 = tro3
        self.Wlim = Wlim
        self.taglim = taglim
        self.tracelim = tracelim

        self.wrule = wrule
        self.tagrule = tagrule

        self.te = Trace(self.Ne, self.tre[0], self.tre[1], self.tracelim)
        self.to = Trace(self.No, self.tro[0], self.tro[1], self.tracelim)

        if self.trm is not None:
            self.tm = Trace(self.No, self.trm[0], self.trm[1], self.tracelim)
        if self.tre2 is not None:
            self.te2 = Trace(self.Ne, self.tre2[0], self.tre2[1], self.tracelim)
        if self.tro2 is not None:
            self.to2 = Trace(self.No, self.tro2[0], self.tro2[1], self.tracelim)
        if self.tro3 is not None:
            self.to3 = Trace(self.No, self.tro3[0], self.tro3[1], self.tracelim)
        
        self.tag = np.zeros((self.No, self.Ne))


    def reset(self):

        super().reset()
        self.te.reset()
        self.to.reset()
        if self.trm is not None:
            self.tm.reset()
        if self.tre2 is not None:
            self.te2.reset()
        if self.tro2 is not None:
            self.to2.reset()
        if self.tro3 is not None:
            self.to3.reset()


    def __call__(self, xe, xm):

        self.xe = self.transform(xe)
        self.xm = xm 
        return self.W @ self.xe


    def update(self, xo, learn=True):

        self.te.update(self.xe)
        if self.tre2 is not None:
            self.te2.update(self.xe)

        self.to.update(xo)

        if self.trm is not None:
            self.tm.update(self.xm)

        if self.tro2 is not None:
            self.to2.update(xo)

        if self.tro3 is not None:
            self.to3.update(xo)

        if learn:

            dW = self.apply_rule(self.wrule, self.xe, xo, xm)
            self.W += dW
            self.W[self.W > self.Wlim] = self.Wlim
            self.W[self.W < -self.Wlim] = -self.Wlim

            if self.tagrule is not None:
                dtag = self.apply_rule(self.tagrule, self.xe, xo, xm)
                self.tag += dtag
                self.tag[self.tag > self.taglim] = self.taglim
                self.tag[self.tag < -self.taglim] = -self.taglim
            

    def apply_rule(self, rule, xe, xo, xm):

        dx = np.zeros((self.No, self.Ne))
        for chunk in rule:
            dchunk = np.ones((self.No, self.Ne))
            for i, value in chunk:
                if i == 0:
                    dchunk = dchunk*(xe + value)

                elif i == 1:
                    dchunk = dchunk*(self.te() + value)
                elif i == 2:
                    dchunk = dchunk*(self.te2() + value)
                elif i == 3:
                    dchunkT = dchunk.T*(xo + value)
                    dchunk = dchunkT.T
                elif i == 4:
                    dchunkT = dchunk.T*(self.to() + value)
                    dchunk = dchunkT.T
                elif i == 5:
                    dchunkT = dchunk.T*(self.to2() + value)
                    dchunk = dchunkT.T
                elif i == 6:
                    dchunkT = dchunk.T*(self.to3() + value)
                    dchunk = dchunkT.T
                elif i == 7:
                    dchunkT = dchunk.T*(xm + value)
                    dchunk = dchunkT.T
                elif i == 8:
                    dchunkT = dchunk.T*(self.tm() + value)
                    dchunk = dchunkT.T
                elif i == 9:
                    dchunk = dchunk*(self.W + value)
                elif i == 10:
                    dchunk *= value
                elif i == 11:
                    dchunk = dchunk*(self.tag + value)
                elif i == 12:
                    dchunk = dchunk*np.sign(self.W + value)
                elif i == 13:
                    dchunk = dchunk*np.sign(value)
                elif i == 14:
                    dchunk = dchunk*np.sign(self.tag + value)
                elif i == 15:
                    dchunk *= value
                else:
                    raise ValueError

            dx += dchunk
        return dx



class ModulatedSTDPrule(LoihiSynapse):
    """Implement a three factor STDP rule with traces

    """

    def __init__(self, Ne, No, W0, tre, tro, trm, Ap, Ad, transform,
        Wlim=1, tracelim=10):

        """
        Parameters
        ----------

        Ne : int
            number of presynaptic neurons
        No : int
            number of postsynaptic neurons
        W0 : np.array
            a 2D array with the initial synaptic weights
        tre : tuple
            presynaptic trace tuple
        tro : tuple 
            postsynaptic trace tuple
        trm : tuple
            modulatory trace tuple
        Ap : float
            potentiation component of STDP
        Ad : float
            depletion component of STDP
        transform : Transform
            apply a synaptic transform to the input
        Wlim : float
            clamping parameter for synaptic weights
        tracelim : float
            clamping parameter for synaptic traces
        
        """

        if trace:
            wrule = [
                [(15,Ap),(3,0),(1,0),(8,0)],
                [(15,-Ad),(0,0),(4,0),(8,0)]
                ]
        else:
            wrule = [
                [(15,Ap),(3,0),(1,0),(7,0)],
                [(15,-Ad),(0,0),(4,0),(7,0)]
                ]


        super().__init__(Ne, No, W0, tre, tro, wrule, transform, Wlim=Wlim, trm=trm, tracelim=tracelim)


class STDPrule(LoihiSynapse):
    """Implement a  simple STDP rule with traces

    """

    def __init__(self, Ne, No, W0, tre, tro, trm, Ap, Ad, transform,
        trace=True, Wlim=1, tracelim=10):

        """
        Parameters
        ----------

        Ne : int
            number of presynaptic neurons
        No : int
            number of postsynaptic neurons
        W0 : np.array
            a 2D array with the initial synaptic weights
        tre : tuple
            presynaptic trace tuple
        tro : tuple 
            postsynaptic trace tuple
        trm : tuple
            modulatory trace tuple
        Ap : float
            potentiation component of STDP
        Ad : float
            depletion component of STDP
        transform : Transform
            apply a synaptic transform to the input
        Wlim : float
            clamping parameter for synaptic weights
        tracelim : float
            clamping parameter for synaptic traces
        
        """

        if trace:
            wrule = [
                [(15,Ap),(3,0),(1,0),(8,0)],
                [(15,-Ad),(0,0),(4,0),(8,0)]
                ]
        else:
            wrule = [
                [(15,Ap),(3,0),(1,0),(7,0)],
                [(15,-Ad),(0,0),(4,0),(7,0)]
                ]


        super().__init__(Ne, No, W0, tre, tro, wrule, transform, Wlim=Wlim, trm=trm, tracelim=tracelim)


class MSErule(LoihiSynapse):
    """Implement a STDP rule

    """

    def __init__(self, Ne, No, W0, tre, tro, trm, lr, trace=True,
        transform=None, Wlim=1, tracelim=10):

        """
        Parameters
        ----------

        Ne : int
            number of presynaptic neurons
        No : int
            number of postsynaptic neurons
        W0 : np.array
            a 2D array with the initial synaptic weights
        tre : tuple
            presynaptic trace tuple
        tro : tuple 
            postsynaptic trace tuple
        trm : tuple
            modulatory trace tuple
        lr : float
            Learning rate
        lowpass : bool
            applies an alpha transform to inputs
        tau : float
            characteristic time of alpha transform
        delay : bool
            if True, applies a 1 timestep delay
        Wlim : float
            clamping parameter for synaptic weights
        tracelim : float
            clamping parameter for synaptic traces
        
        """

        if trace:
            wrule = [
                [(15,lr),(3,0),(1,0),(8,0)],
                [(15,-lr),(0,0),(4,0),(8,0)]
                ]
        else:
            wrule = [
                [(15,lr),(1,0),(8,0)],
                [(15,-lr),(1,0),(4,0)]
                ]

        super().__init__(Ne, No, W0, tre, tro, wrule, transform,
            Wlim=Wlim, trm=trm, tracelim=tracelim)


