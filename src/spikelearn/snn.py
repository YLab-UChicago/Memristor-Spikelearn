
class SpikingNet:

    def __init__(self):

        self.layers = {}
        self.inputs = []
        self.outputs = []
        self.synapses = []
        self.pre_synapse = {}
        self.pos_synapse = {}
        self.layer_synapses = {}

    def add_layer(self, snl, name):
        self.layers[name] = snl
        self.layer_synapses[name] = []

    def add_input(self, name):
        self.inputs.append(name)

    def add_output(self, name):
        self.outputs.append(name)

    def add_synapse(self, pos_name, syn, *pre_names):
        syn_ind = len(self.synapses)
        self.synapses.append(syn)
        self.pre_synapse[syn_ind] = list(pre_names)
        self.pos_synapse[syn_ind] = pos_name
        self.layer_synapses[pos_name].append(syn_ind)

    def reset(self):
        """Resets all elements of the network
        
        Broadcasts a reset signal to all nodes and synapses
        in the network
        """
        for _ ,layer in self.layers.items():
            layer.reset()
        for syn in self.synapses:
            syn.reset()

    def __call__(self, *args, learn=True):
        self.forward(*args, learn=learn)
        self._update()
        return [self.layers[name].s for name in self.outputs]
    
    def forward(self, *args, learn=True):
        self.learn=learn
        if len(args) != len(self.inputs):
            raise ValueError("Wrong number of inputs in snn")
        i_dict = {name:a for name, a in zip(self.inputs, args)}
        out_synapse  = []
        for i, syn in enumerate(self.synapses):
            ul = []
            for c in self.pre_synapse[i]:
                if c in i_dict.keys():
                    ul.append(i_dict[c])
                else:
                    ul.append(self.layers[c].s)
            out_synapse.append(syn(*ul))
        for name, neuron in self.layers.items():
            xl = [out_synapse[ind] for ind in self.layer_synapses[name]]
            neuron(*xl)

    def update(self):
        self._update()
    def get_output(self):
        return [self.layers[name].s for name in self.outputs]

    def _update(self):
        for ind, syn in enumerate(self.synapses):
            syn.update(self.layers[self.pos_synapse[ind]].s, self.learn)



